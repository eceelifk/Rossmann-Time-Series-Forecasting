import nbformat

with open('rossmann_sales_forecast.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # 1. Change epochs from 7 to 10
        if 'epochs=7' in cell.source:
            cell.source = cell.source.replace('epochs=7', 'epochs=10')
            
        # 2. Fix the IndexError in safe_mape by flattening arrays
        if 'lstm_preds = evaluate_model' in cell.source:
            # We want to flatten all these to 1D to prevent masking issues
            cell.source = cell.source.replace(
                "y_true = scaler.inverse_transform(y_test.reshape(-1, 1))",
                "y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()"
            )
            cell.source = cell.source.replace(
                "lstm_preds = evaluate_model(lstm_model, test_loader)",
                "lstm_preds = evaluate_model(lstm_model, test_loader).flatten()"
            )
            cell.source = cell.source.replace(
                "gru_preds = evaluate_model(gru_model, test_loader)",
                "gru_preds = evaluate_model(gru_model, test_loader).flatten()"
            )
            
        # 3. If there is xgb_preds, ensure it's flattened (it already has .flatten() but let's be safe)
        # Actually it's already flattened, the issue was y_true wasn't flattened.

with open('rossmann_sales_forecast.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Fixed array shapes and updated epochs to 10.")
