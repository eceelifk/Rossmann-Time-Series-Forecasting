import nbformat

nb_path = 'rossmann_sales_forecast.ipynb'
try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Replace cell 9 with the proper LSTM training code
    new_source_9 = '''print("\\n--- LSTM MODEL EGITIMI (Lutfen Bekleyin) ---")
lstm_model = SalesLSTM()
lstm_model, lstm_time = train_model(lstm_model, train_loader, epochs=5)
'''
    nb.cells[9].source = new_source_9

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print('Fixed cell 9!')
except Exception as e:
    print(e)
