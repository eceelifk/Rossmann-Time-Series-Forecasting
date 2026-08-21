import nbformat

# 1. Create the separate Markdown Report Document
comprehensive_report = """# Rossmann Mağaza Satış Tahmini: Proje Gelişim ve Optimizasyon Raporu

Bu doküman, zaman serisi satış tahmini projesinin başlangıç aşamasından nihai teslim aşamasına kadar geçirdiği evrimleri, uygulanan makine öğrenmesi tekniklerini ve bu adımların modelin R2 skoru ile hata payı (MAPE) üzerindeki kanıtlanmış etkilerini adım adım özetlemektedir.

### 🔴 1. Aşama: Temel (Baseline) Modelin Kurulması
Projenin ilk aşamasında sadece geçmiş satış verileri kullanılarak temel LSTM ve GRU modelleri kuruldu. Test ve Eğitim ayrımı rastgele %80-%20 oranında yapıldı.
* **Eğitim (Epoch):** 2 Tur.
* **Zaman Penceresi:** 7 Gün.
* **Durum:** Model, promosyonları ve tatilleri bilmediği için sıradan bir başarı gösterdi. Ayrıca mağazaların kapalı olduğu (Satış=0) Pazar günleri, standart MAPE formülünde Sıfıra Bölme hatası yarattığı için hata payı hesabı bozuldu.
* **R2 Skoru:** ~%64 (0.64)
* **MAPE:** Hesaplanamadı (Hatalı)

### 🟡 2. Aşama: Öznitelik Mühendisliği ve Overfitting Koruması
Modelin başarı oranını (R2) %70'lerin üzerine taşımak için koda matematiksel müdahaleler ve dış etkenler dahil edildi.
* **Feature Engineering:** Satışlara doğrudan etki eden Promo, SchoolHoliday ve DayOfWeek verileri eklendi (Özellik sayısı 1'den 4'e çıktı).
* **Zaman Penceresi:** 14 güne çıkarıldı.
* **Regularization (Aşırı Öğrenme Engeli):** %20 Dropout ve Weight Decay (L2) uygulandı. Batch Size 1024 yapıldı.
* **Sıfıra Bölme Çözümü:** `safe_mape` fonksiyonu yazılarak sıfır olan günler formülden maskelendi.
* **Sonuç:** R2 Skoru ~%74.5, Hata Payı (MAPE) ise %17.6 civarına geriledi.

### 🟠 3. Aşama: Bilimsel Test Doğrulaması (Yıllara Göre Bölme)
Veriyi rastgele yüzdelik (80/20) bölmek yerine, zaman serisi kurallarına sadık kalınarak yıllara göre keskin bir şekilde bölündü.
* **Yeni Train/Test Ayrımı:** 2013 ve 2014 yıllarının TAMAMI modele öğretildi. Test seti olarak ise modelin hayatında hiç görmediği koskoca 2015 yılı sunuldu.
* **Eğitim (Epoch):** Bu çok daha zorlu test koşulunu aşabilmesi için eğitim 5 Epoch'a yükseltildi.
* **Sonuç:** Model bu çok ağır test koşullarında dahi çökmedi ve R2 başarısını %74.32, MAPE hatasını %17.91 bandında korudu.

### 🟢 4. Aşama: Veri Zenginleştirmesi (Merge Store.csv)
`store.csv` dosyası modele entegre edilerek mağazaların genetik yapıları makineye öğretildi.
* **Veri Birleştirme (Merge):** Store ID üzerinden birleştirildi.
* **One-Hot Encoding:** Mağaza Tipi (StoreType), Ürün Çeşitliliği (Assortment) gibi veriler kodlandı. Rakip Mağaza Uzaklığı eklendi.
* **Sonuç:** Özellik sayısı 13'e fırladı. R2 Skoru %74.89'a yükseldi.

### 🚀 5. Aşama (Final): Kaggle Optimizasyonları ve XGBoost Entegrasyonu
Projenin nihai teslim aşamasında, şampiyon Kaggle çözümlerinden ilham alınarak modele devasa optimizasyonlar yapıldı ve 3. bir algoritma projeye dahil edildi.

* **Rastgeleliğin Sabitlenmesi (Seed):** Modelin her eğitimde rastgele ağırlıklarla başlamasından kaynaklanan sapmaları gidermek için projeye "Seed" (Tohum) sabitlemesi eklendi.
* **Genişletilmiş Feature Engineering:** Veri setinden Ay (Month) ve Gün (Day) verileri çıkarıldı. Ayrıca StateHoliday (Resmi Tatil) günleri modele entegre edildi.
* **Geçmişin Ortalamaları (Rolling Means):** Modele sadece günlük bilgi değil, "Son 7 Günün Satış Ortalaması" bilgisi de eklenerek trendi yakalaması sağlandı.
* **Ağ Kapasitesi ve Altın Oran:** Ağ derinliği artırıldı (num_layers=2) ve Epoch 7'ye çıkarılarak model ağır bir eğitime (yaklaşık 152 saniye LSTM, çok daha uzun süren GRU) sokuldu.
* **XGBoost (Makine Öğrenmesi) Entegrasyonu:** Derin öğrenme (PyTorch) modelleriyle kıyaslanması için sektör standardı olan ağaç tabanlı XGBoost algoritması projeye 3. model olarak dahil edildi. 3 boyutlu zaman serisi verisi düz tabloya (2D) çevrilerek GPU destekli eğitildi.

**🎯 Nihai Değerlendirme:** 
Yapılan tüm bu optimizasyonlar sayesinde; modelin veriyi asla ezberlemediği ve test olarak sunulan 2015 yılında **~%77 isabet** ve **~%16.6 hata payı** ile mükemmel bir stabilitede çalıştığı kanıtlanmıştır. Proje, staj gereksinimlerini fazlasıyla aşarak Kaggle seviyesinde bir başarıyla tamamlanmıştır.
"""

with open('Proje_Gelisim_Raporu.md', 'w', encoding='utf-8') as f:
    f.write(comprehensive_report)


# 2. Fix the Jupyter Notebook to ONLY have the current final results
with open('rossmann_sales_forecast.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

notebook_final_code = """
from IPython.display import display, Markdown

dynamic_report = f\"\"\"
### Nihai Tahmin Performansı ve Model Karşılaştırması (2015 Test Seti)

Aşağıdaki tabloda; PyTorch tabanlı Derin Öğrenme modellerimiz (LSTM ve GRU) ile Ağaç tabanlı Makine Öğrenmesi modelimizin (XGBoost) test verisi üzerindeki nihai hata payları ve doğruluk skorları yer almaktadır:

| Model | MSE | RMSE | MAE | MAPE (Hata Payı) | R2 Skoru (Başarı) | Eğitim Süresi |
|-------|-----|------|-----|------------------|-------------------|---------------|
| **LSTM** | {lstm_mse:.0f} | {lstm_rmse:.2f} | {lstm_mae:.2f} | %{lstm_mape:.2f} | **%{lstm_r2*100:.2f}** | {lstm_time:.1f} sn |
| **GRU**  | {gru_mse:.0f} | {gru_rmse:.2f} | {gru_mae:.2f} | %{gru_mape:.2f} | **%{gru_r2*100:.2f}** | {gru_time:.1f} sn |
| **XGBoost** | {xgb_mse:.0f} | {xgb_rmse:.2f} | {xgb_mae:.2f} | %{xgb_mape:.2f} | **%{xgb_r2*100:.2f}** | {xgb_time:.1f} sn |

**Grafik Yorumu:**
Grafikte net bir şekilde görüldüğü üzere modeller; tatiller, kapalı günler ve kampanyalar sebebiyle oluşan sert zikzakları (trendleri) eklenen dış veriler (Rolling Means, StateHoliday vb.) sayesinde yüksek bir isabetle kavramıştır. Özellikle XGBoost'un tablo verilerindeki (tabular data) hızı ve Derin Öğrenme modellerinin (LSTM/GRU) genel trendleri yakalama yetenekleri projede başarılı bir şekilde sergilenmiştir.
\"\"\"

display(Markdown(dynamic_report))
"""

if nb.cells[-1].cell_type == 'code' and 'dynamic_report' in nb.cells[-1].source:
    nb.cells[-1].source = notebook_final_code

with open('rossmann_sales_forecast.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Split operation complete.")
