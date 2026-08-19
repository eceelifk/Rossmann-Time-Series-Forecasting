import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Aşama 2: PyTorch ile Zaman Serisi Satış Tahmini (LSTM vs GRU)\n",
                "\n",
                "Bu projede Rossmann Mağaza Satışları veri setini kullanarak gelecekteki mağaza satışlarını (regresyon problemi) tahmin ediyoruz.\n",
                "Train/Test ayrımı %80 Eğitim, %20 Test olacak şekilde güncellenmiştir."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "from torch.utils.data import TensorDataset, DataLoader\n",
                "from sklearn.preprocessing import MinMaxScaler\n",
                "import matplotlib.pyplot as plt\n",
                "from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error\n",
                "import time\n",
                "\n",
                "# PyTorch için device ayarı\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f'Kullanılan cihaz: {device}')"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Veri Yükleme ve Ön İşleme"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Veriyi yükleme\n",
                "df = pd.read_csv('train.csv', low_memory=False)\n",
                "\n",
                "# Sadece açık olan mağazaları filtreleme\n",
                "df = df[df['Open'] == 1]\n",
                "\n",
                "# Tarihi objeye çevirme ve mağaza/tarih sırasına koyma\n",
                "df['Date'] = pd.to_datetime(df['Date'])\n",
                "df = df.sort_values(by=['Store', 'Date'])\n"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Kayan Pencere (Sliding Window) ve Veri Ölçekleme\n",
                "Veri seti %80 Eğitim (Train) ve %20 Test olarak bölünecek şekilde ayarlanmıştır."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "scaler = MinMaxScaler()\n",
                "df['Sales_Scaled'] = scaler.fit_transform(df[['Sales']])\n",
                "\n",
                "def create_sequences(data_grouped, seq_length):\n",
                "    xs, ys = [], []\n",
                "    for store_id, group in data_grouped:\n",
                "        store_sales = group['Sales_Scaled'].values\n",
                "        if len(store_sales) <= seq_length:\n",
                "            continue\n",
                "        for i in range(len(store_sales) - seq_length):\n",
                "            xs.append(store_sales[i:(i + seq_length)])\n",
                "            ys.append(store_sales[i + seq_length])\n",
                "    return np.array(xs), np.array(ys)\n",
                "\n",
                "# Toplam veri ~31 ay. %80'i ~25 ay yapar. \n",
                "# Başlangıç: 2013-01. %80'lik kesim tarihi: 2015-02-01\n",
                "split_date = '2015-02-01'\n",
                "train_df = df[df['Date'] < split_date]\n",
                "test_df = df[df['Date'] >= split_date]\n",
                "\n",
                "seq_length = 7\n",
                "print('Tensörler oluşturuluyor...')\n",
                "\n",
                "X_train, y_train = create_sequences(train_df.groupby('Store'), seq_length)\n",
                "X_test, y_test = create_sequences(test_df.groupby('Store'), seq_length)\n",
                "\n",
                "X_train = np.expand_dims(X_train, axis=-1)\n",
                "X_test = np.expand_dims(X_test, axis=-1)\n",
                "\n",
                "print('Eğitim seti (%80):', X_train.shape, y_train.shape)\n",
                "print('Test seti (%20):', X_test.shape, y_test.shape)"
            ],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "batch_size = 2048 \n",
                "X_train_t = torch.tensor(X_train, dtype=torch.float32)\n",
                "y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)\n",
                "X_test_t = torch.tensor(X_test, dtype=torch.float32)\n",
                "y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)\n",
                "\n",
                "train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)\n",
                "test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)"
            ],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "class SalesLSTM(nn.Module):\n",
                "    def __init__(self, input_size=1, hidden_size=64, num_layers=1):\n",
                "        super(SalesLSTM, self).__init__()\n",
                "        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)\n",
                "        self.fc = nn.Linear(hidden_size, 1)\n",
                "        \n",
                "    def forward(self, x):\n",
                "        out, _ = self.lstm(x)\n",
                "        out = self.fc(out[:, -1, :])\n",
                "        return out\n",
                "\n",
                "class SalesGRU(nn.Module):\n",
                "    def __init__(self, input_size=1, hidden_size=64, num_layers=1):\n",
                "        super(SalesGRU, self).__init__()\n",
                "        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)\n",
                "        self.fc = nn.Linear(hidden_size, 1)\n",
                "        \n",
                "    def forward(self, x):\n",
                "        out, _ = self.gru(x)\n",
                "        out = self.fc(out[:, -1, :])\n",
                "        return out"
            ],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "def train_model(model, train_loader, epochs=2, lr=0.001):\n",
                "    model = model.to(device)\n",
                "    criterion = nn.MSELoss()\n",
                "    optimizer = torch.optim.Adam(model.parameters(), lr=lr)\n",
                "    start_time = time.time()\n",
                "    for epoch in range(epochs):\n",
                "        model.train()\n",
                "        epoch_loss = 0\n",
                "        for batch_x, batch_y in train_loader:\n",
                "            batch_x, batch_y = batch_x.to(device), batch_y.to(device)\n",
                "            optimizer.zero_grad()\n",
                "            outputs = model(batch_x)\n",
                "            loss = criterion(outputs, batch_y)\n",
                "            loss.backward()\n",
                "            optimizer.step()\n",
                "            epoch_loss += loss.item()\n",
                "        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.6f}')\n",
                "    end_time = time.time()\n",
                "    training_time = end_time - start_time\n",
                "    print(f'Eğitim tamamlandı! Süre: {training_time:.2f} sn')\n",
                "    return model, training_time"
            ],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "print(\"--- LSTM MODEL EĞİTİMİ (Lütfen Bekleyin) ---\")\n",
                "lstm_model = SalesLSTM()\n",
                "lstm_model, lstm_time = train_model(lstm_model, train_loader, epochs=2)"
            ],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "print(\"\\n--- GRU MODEL EĞİTİMİ (Lütfen Bekleyin) ---\")\n",
                "gru_model = SalesGRU()\n",
                "gru_model, gru_time = train_model(gru_model, train_loader, epochs=2)"
            ],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "def evaluate_model(model, test_loader):\n",
                "    model.eval()\n",
                "    predictions = []\n",
                "    with torch.no_grad():\n",
                "        for batch_x, _ in test_loader:\n",
                "            batch_x = batch_x.to(device)\n",
                "            outputs = model(batch_x)\n",
                "            predictions.extend(outputs.cpu().numpy())\n",
                "    \n",
                "    predictions = scaler.inverse_transform(predictions)\n",
                "    return predictions\n",
                "\n",
                "lstm_preds = evaluate_model(lstm_model, test_loader)\n",
                "gru_preds = evaluate_model(gru_model, test_loader)\n",
                "y_true = scaler.inverse_transform(y_test.reshape(-1, 1))\n",
                "\n",
                "lstm_mse = mean_squared_error(y_true, lstm_preds)\n",
                "lstm_rmse = np.sqrt(lstm_mse)\n",
                "lstm_mape = mean_absolute_percentage_error(y_true, lstm_preds) * 100\n",
                "\n",
                "gru_mse = mean_squared_error(y_true, gru_preds)\n",
                "gru_rmse = np.sqrt(gru_mse)\n",
                "gru_mape = mean_absolute_percentage_error(y_true, gru_preds) * 100\n",
                "\n",
                "print(\"=\"*50)\n",
                "print(\"METRİK KARŞILAŞTIRMASI (%80 Train, %20 Test)\")\n",
                "print(\"=\"*50)\n",
                "print(f'LSTM -> MSE: {lstm_mse:.0f} | RMSE: {lstm_rmse:.2f} | MAPE: %{lstm_mape:.2f} (Süre: {lstm_time:.1f} sn)')\n",
                "print(f'GRU  -> MSE: {gru_mse:.0f} | RMSE: {gru_rmse:.2f} | MAPE: %{gru_mape:.2f} (Süre: {gru_time:.1f} sn)')\n",
                "print(\"=\"*50)"
            ],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "plt.figure(figsize=(15, 6))\n",
                "plt.plot(y_true[:100], label='Gerçek Satışlar', marker='o', linewidth=2)\n",
                "plt.plot(lstm_preds[:100], label='LSTM Tahminleri', marker='x', alpha=0.8)\n",
                "plt.plot(gru_preds[:100], label='GRU Tahminleri', marker='^', alpha=0.8)\n",
                "plt.title('Zaman Serisi Tahmini (İlk 100 Gün Karşılaştırması)')\n",
                "plt.xlabel('Zaman Adımı (Gün)')\n",
                "plt.ylabel('Satış Miktarı')\n",
                "plt.legend()\n",
                "plt.grid(True)\n",
                "plt.show()"
            ],
            "outputs": []
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

target_path = r'C:\Users\elife\Desktop\mağazasatıştahmin\rossmann_sales_forecast.ipynb'
with open(target_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4)
