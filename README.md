# 🛒 Rossmann Mağaza Satış Tahmini Modeli

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-%23150458.svg?style=for-the-badge&logo=XGBoost&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Microsoft Staj Programı - 2. Aşama Projesi** kapsamında geliştirilmiş, derin öğrenme (Deep Learning) ve makine öğrenmesi (Machine Learning) tabanlı zaman serisi satış tahminleme projesidir.

---

##  Proje Hakkında
Bu projenin amacı, Avrupa'nın önde gelen eczane zincirlerinden Rossmann'ın tarihsel satış verilerini kullanarak **gelecekteki günlük mağaza satışlarını yüksek doğrulukla tahmin etmektir.** 

Klasik yaklaşımlardan farklı olarak sadece tek bir mağazanın değil, yüzlerce mağazanın verisi entegre bir şekilde işlenmiş; zaman serisi dinamikleri **Kayan Pencere (Sliding Window)** yaklaşımıyla PyTorch tensörlerine uyarlanarak modellenmiştir.

##  Öne Çıkan Özellikler ve Veri Mühendisliği
Modelin ezberlemesini (overfitting) önlemek ve yüksek skorlara ulaşmasını sağlamak için yoğun bir **Feature Engineering** (Veri Mühendisliği) çalışması yapılmıştır:
- **Tarihsel Zenginleştirme:** Hafta içi/hafta sonu ayrımları, okul ve resmi tatil günleri (StateHoliday) modele entegre edildi.
- **Rakip Mağaza Etkisi:** `CompetitionDistance` (Rakip mağazaya uzaklık) metrikleri NaN boşluklarından arındırılarak eklendi.
- **Hareketli Ortalamalar:** Modelin anlık zıplamalardan etkilenmemesi için `Sales_Rolling_7` (7 Günlük Satış Ortalaması) hesaplandı.
- **Gelişmiş Ölçeklendirme:** Hem satış verileri hem de rakip mesafe verileri `MinMaxScaler` kullanılarak optimize edildi.

##  Model Mimarileri
Projeye birbirine alternatif 3 farklı güçlü algoritma dahil edilmiş ve performansları bilimsel metriklerle karşılaştırılmıştır:
1. **LSTM (Long Short-Term Memory):** 256 Hidden Size ve %30 Dropout ile optimize edilmiş derin öğrenme ağı.
2. **GRU (Gated Recurrent Unit):** 256 Hidden Size'lı, daha hızlı eğitim süresine sahip RNN türevi.
3. **XGBoost (Extreme Gradient Boosting):** 1000 ağaç (n_estimators), 9 max_depth ve 0.9 subsample ayarlarıyla modifiye edilmiş, şampiyon makine öğrenmesi algoritması.

---

##  Nihai Performans ve Sonuçlar
Modeller, **2013-2014 verileriyle eğitilmiş** ve daha önce hiç görmedikleri **2015 test verisi (%20)** üzerinde sınanmıştır. 

Aşağıdaki tablo, modellerin test seti üzerindeki nihai (gerçek dünya) başarısını göstermektedir:

| Model | MSE | RMSE | MAE | MAPE (%) | R2 Skoru | Eğitim Süresi |
|-------|-----|------|-----|----------|----------|---------------|
| **XGBoost** | ~784,000 | ~885.00 | ~620.00 | **%9.45** | **%91.66** | ~110 sn |
| **LSTM** | ~2,070,000| ~1438.00| ~1010.00| **%16.80**| **%77.98** | ~22 dk |
| **GRU** | ~2,260,000| ~1503.00| ~1050.00| **%17.50**| **%75.95** | ~18 dk |

*(Yukarıdaki R2 Skorları, hedeflenen %80 doğruluk barajını XGBoost ile %91'lere taşıyarak projenin beklentileri fazlasıyla aştığını kanıtlamaktadır.)*

### Modellerin Metrik Karşılaştırması
Aşağıdaki grafikler, modellerin hata paylarını ve doğruluk oranlarını görsel olarak kıyaslamaktadır:

![Metrik Karşılaştırması](metrics_comparison.png)

---

##  Kurulum ve Kullanım
Bu projeyi kendi bilgisayarınızda çalıştırmak için:

**1. Gerekli kütüphaneleri yükleyin:**
```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn xgboost
```

**2. Jupyter Notebook'u başlatın:**
```bash
jupyter notebook
```

**3. Test Edin:**
`rossmann_sales_forecast.ipynb` dosyasını açıp üst menüden **Kernel -> Restart & Run All** seçeneğine tıklayarak baştan sona veri işleme, eğitim, test ve metrik çizim süreçlerini canlı olarak deneyimleyebilirsiniz.
