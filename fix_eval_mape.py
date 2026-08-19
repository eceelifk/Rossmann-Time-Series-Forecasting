import nbformat

nb_path = 'rossmann_sales_forecast.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            if 'evaluate_model' in cell.source and 'results_df' in cell.source:
                new_source = '''from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import torch

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
    
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Sifira bolme hatasini engelleyen guvenli MAPE fonksiyonu
def safe_mape(y_true, y_pred):
    mask = y_true != 0
    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100

lstm_preds = evaluate_model(lstm_model, test_loader)
gru_preds = evaluate_model(gru_model, test_loader)
y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

# LSTM Metrikleri
lstm_mse = mean_squared_error(y_true, lstm_preds)
lstm_rmse = np.sqrt(lstm_mse)
lstm_mae = mean_absolute_error(y_true, lstm_preds)
lstm_r2 = r2_score(y_true, lstm_preds)
lstm_mape = safe_mape(y_true, lstm_preds)

# GRU Metrikleri
gru_mse = mean_squared_error(y_true, gru_preds)
gru_rmse = np.sqrt(gru_mse)
gru_mae = mean_absolute_error(y_true, gru_preds)
gru_r2 = r2_score(y_true, gru_preds)
gru_mape = safe_mape(y_true, gru_preds)

results_df = pd.DataFrame({
    'Model': ['LSTM', 'GRU'],
    'MSE': [lstm_mse, gru_mse],
    'RMSE': [lstm_rmse, gru_rmse],
    'MAE': [lstm_mae, gru_mae],
    'MAPE (%)': [lstm_mape, gru_mape],
    'R2 Skoru': [lstm_r2, gru_r2],
    'Eğitim Süresi (sn)': [lstm_time, gru_time]
})

print("\\n--- LSTM ve GRU Modellerinin Karşılaştırması ---")
print(results_df.to_string(index=False))
'''
                nb.cells[i].source = new_source
                print('Updated evaluation cell with MAPE.')

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
except Exception as e:
    print("Error:", e)
