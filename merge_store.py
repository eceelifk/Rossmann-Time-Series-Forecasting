import nbformat

nb_path = 'rossmann_sales_forecast.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 1. Update Data Loading and Preprocessing Cell
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and "pd.read_csv('train.csv'" in cell.source:
            new_source = '''import pandas as pd
import numpy as np

# Veriyi yükleme
df = pd.read_csv('train.csv', low_memory=False)
store_df = pd.read_csv('store.csv', low_memory=False)

# Sadece açık olan mağazaları filtreleme (Kapalıysa satış zaten 0'dır)
df = df[df['Open'] == 1]

# Store verisini ana veri setine ekliyoruz (MERGE)
df = df.merge(store_df, on='Store', how='left')

# Tarih sütununu datetime formatına çevirme
df['Date'] = pd.to_datetime(df['Date'])

# Tarihe ve mağazaya göre sıralama
df = df.sort_values(by=['Store', 'Date'])
'''
            nb.cells[i].source = new_source
            print('Updated Data Loading cell.')

        # 2. Update Feature Engineering Cell
        elif cell.cell_type == 'code' and 'DayOfWeek_Scaled' in cell.source and 'scaler' in cell.source:
            new_source = '''from sklearn.preprocessing import MinMaxScaler

# 1. Mevcut öznitelikler
df['DayOfWeek_Scaled'] = (df['DayOfWeek'] - 1) / 6.0
df['Promo'] = df['Promo'].astype(float)
df['SchoolHoliday'] = df['SchoolHoliday'].astype(float)

# 2. Yeni eklenen Store özellikleri
# CompetitionDistance (Rakip uzaklığı): Boş olanları medyan ile dolduralım
df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())

# StoreType ve Assortment (Metinsel/Kategorik veriler) One-Hot Encoding ile 0 ve 1'lere çevrilir
df = pd.get_dummies(df, columns=['StoreType', 'Assortment'], dtype=float)

# Hangi yeni kategorik sütunların oluştuğunu bulalım (örn: StoreType_a, Assortment_c vb.)
store_type_cols = [c for c in df.columns if 'StoreType_' in c]
assortment_cols = [c for c in df.columns if 'Assortment_' in c]

# Promo2 zaten 0 ve 1'den oluşuyor
df['Promo2'] = df['Promo2'].astype(float)

# Satışları ve Rakip Uzaklığını Ölçeklendirme
scaler = MinMaxScaler()
df['Sales_Scaled'] = scaler.fit_transform(df[['Sales']])

dist_scaler = MinMaxScaler()
df['CompetitionDistance_Scaled'] = dist_scaler.fit_transform(df[['CompetitionDistance']])

# Tüm Özellikleri Birleştirme
scaled_features = ['Sales_Scaled', 'Promo', 'DayOfWeek_Scaled', 'SchoolHoliday', 'CompetitionDistance_Scaled', 'Promo2'] + store_type_cols + assortment_cols

def create_sequences(data_grouped, seq_length):
    xs, ys = [], []
    for store_id, group in data_grouped:
        store_data = group[scaled_features].values
        if len(store_data) <= seq_length:
            continue
        for i in range(len(store_data) - seq_length):
            xs.append(store_data[i:(i + seq_length)])
            ys.append(store_data[i + seq_length, 0]) # Sales_Scaled her zaman 0. indekste
    return np.array(xs), np.array(ys)

# Veriyi 2013-2014 Eğitim, 2015 Test olacak şekilde ayırıyoruz
split_date = pd.to_datetime('2015-01-02')

train_df = df[df['Date'] < split_date]
test_df = df[df['Date'] >= split_date]

# Zaman penceresi
seq_length = 14
print('Tensörler oluşturuluyor... (Birkaç dakika sürebilir)')

X_train, y_train = create_sequences(train_df.groupby('Store'), seq_length)
X_test, y_test = create_sequences(test_df.groupby('Store'), seq_length)

print('Eğitim seti (%80):', X_train.shape, y_train.shape)
print('Test seti (%20):', X_test.shape, y_test.shape)
print('Kullanılan Özellik Sayısı:', len(scaled_features))
input_size_dynamic = len(scaled_features)
'''
            nb.cells[i].source = new_source
            print('Updated Feature Engineering cell.')

        # 3. Update Model Architectures to use dynamic input_size
        elif cell.cell_type == 'code' and 'SalesLSTM' in cell.source:
            new_source = '''import torch.nn as nn

class SalesLSTM(nn.Module):
    def __init__(self, input_size=input_size_dynamic, hidden_size=64, num_layers=1):
        super(SalesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class SalesGRU(nn.Module):
    def __init__(self, input_size=input_size_dynamic, hidden_size=64, num_layers=1):
        super(SalesGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
'''
            nb.cells[i].source = new_source
            print('Updated Model Classes cell.')

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

except Exception as e:
    print('Error:', e)
