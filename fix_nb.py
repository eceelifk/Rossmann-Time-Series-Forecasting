import nbformat

nb_path = 'rossmann_sales_forecast.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    target_idx = -1
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            if 'split_date' in cell.source and 'train_df' in cell.source:
                new_source = '''scaler = MinMaxScaler()
df['Sales_Scaled'] = scaler.fit_transform(df[['Sales']])

def create_sequences(data_grouped, seq_length):
    xs, ys = [], []
    for store_id, group in data_grouped:
        store_sales = group['Sales_Scaled'].values
        if len(store_sales) <= seq_length:
            continue
        for i in range(len(store_sales) - seq_length):
            xs.append(store_sales[i:(i + seq_length)])
            ys.append(store_sales[i + seq_length])
    return np.array(xs), np.array(ys)

# Veriyi zamana göre tam %80 Eğitim, %20 Test olacak şekilde ayıralım
unique_dates = df['Date'].sort_values().unique()
split_idx = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_idx]

train_df = df[df['Date'] < split_date]
test_df = df[df['Date'] >= split_date]

seq_length = 7
print('Tensörler oluşturuluyor...')

X_train, y_train = create_sequences(train_df.groupby('Store'), seq_length)
X_test, y_test = create_sequences(test_df.groupby('Store'), seq_length)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print('Eğitim seti (%80):', X_train.shape, y_train.shape)
print('Test seti (%20):', X_test.shape, y_test.shape)'''
                nb.cells[i].source = new_source
                print('Updated split cell.')
                
            elif 'plt.show()' in cell.source:
                # We want to add a markdown cell after this graph
                target_idx = i

    # Insert markdown cell after the graph
    comment_md = nbformat.v4.new_markdown_cell(source='''### Grafik Yorumu
Yukarıdaki grafikte ilk 100 günlük gerçek satışlar ile LSTM ve GRU modellerinin tahminleri karşılaştırılmıştır.

* **Trendi Yakalama:** Her iki model de satışlardaki genel iniş-çıkış (mevsimsel) döngülerini başarıyla kavramıştır. 
* **Uç Değerler (Aşırı Yüksek/Düşük Satışlar):** Ani yükseliş yaşanan günlerde (örneğin ~40. gün) modeller tam satış rakamına erişemese de satışın artacağını doğru yönde öngörmüşlerdir.
* **LSTM vs GRU:** Yeşil renkle gösterilen GRU modelinin çizgisinin mavi renkli gerçek satış çizgisine turuncu LSTM çizgisine kıyasla biraz daha yakınsadığını görebiliriz. Bu da R2, RMSE ve MAE metriklerindeki ufak üstünlüğünü doğrulamaktadır.''')

    if target_idx != -1:
        # Only insert if it doesn't already exist
        already_exists = False
        if target_idx + 1 < len(nb.cells):
            if nb.cells[target_idx + 1].cell_type == 'markdown' and 'Grafik Yorumu' in nb.cells[target_idx + 1].source:
                already_exists = True

        if not already_exists:
            nb.cells.insert(target_idx + 1, comment_md)
            print('Inserted graph comment.')

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
except Exception as e:
    print("Error:", e)
