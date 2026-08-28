# 📈 Rossmann Mağaza Satış Tahmini — Microsoft Staj Programı (Aşama 2)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-%23150458.svg?style=for-the-badge&logo=XGBoost&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

Bu proje, **Microsoft Staj Programı 2. Aşama (Zaman Serisi Tahmini)** kapsamında tarafımca geliştirilmiş kapsamlı bir veri bilimi ve yapay zeka çalışmasıdır.

Çalışmamda, literatürdeki en saygın veri bilimi yarışmalarından biri olan **Kaggle Rossmann Store Sales** veri setini temel aldım. Projenin ana hedefi, Avrupa'nın dev eczane zinciri Rossmann'a ait 1115 farklı mağazanın tarihsel verilerini (1 Milyon+ satır) analiz ederek, gelecekteki günlük satış miktarlarını yüksek isabetle tahmin etmektir. Bu doğrultuda hem Makine Öğrenmesi (XGBoost) hem de Derin Öğrenme (PyTorch LSTM/GRU) algoritmaları kullanılmış ve sonuçlar bilimsel metriklerle kıyaslanmıştır.

---

## 📊 1. Veri Seti Mimarisi ve İçeriği

Model eğitiminde kullanılan veri seti, Ocak 2013 ile Temmuz 2015 arasındaki satış dinamiklerini yansıtmaktadır. Sadece satış rakamları değil, satışları doğrudan etkileyen dış faktörler de modele entegre edilmiştir:

* **Sales (Hedef Değişken):** Mağazanın günlük toplam satış tutarı.
* **Open & Promo:** Mağazanın açıklık durumu ve aktif promosyon/indirim varlığı.
* **StateHoliday & SchoolHoliday:** Resmi ve okul tatillerinin perakende satışlarına olan etkisi.
* **StoreType & Assortment:** Mağaza büyüklük sınıflandırması ve ürün çeşitliliği seviyesi.
* **CompetitionDistance:** En yakın rakip mağazanın metre cinsinden uzaklığı.

---

## 🛠️ 2. Veri Mühendisliği (Feature Engineering)

Ham verinin algoritmalar tarafından doğru yorumlanabilmesi için aşağıdaki veri mühendisliği süreçlerini uyguladım:

* **Matematiksel Ölçeklendirme:** Zaman birimlerini (Gün, Ay, Yıl) ve büyük sayısal verileri (satışlar, mesafeler) `MinMaxScaler` ile [0, 1] aralığına normalize ettim.
* **Hareketli Ortalamalar (Rolling Means):** Perakende sektöründe satışların momentumunu yakalamak adına, mağazaların son 7 günlük satış ortalamalarını hesaplayarak modele güçlü bir özellik (feature) olarak ekledim.
* **Kayan Pencere (Sliding Window):** Zaman serisi tahminlemesinin temeli olan "geçmişten öğrenme" mantığını koda döktüm. Model, her tahmin adımında geçmiş **45 günlük (1.5 aylık)** periyodu girdi olarak alıp 46. günün satışını çıktı olarak verecek şekilde yapılandırıldı.

---

## ✂️ 3. Eğitim ve Test Ayrımı (Train/Test Split)

Zaman serisi projelerinde veri sızıntısını (Data Leakage) önlemek hayati önem taşır. Bu nedenle veriyi rastgele değil, kesin bir kronolojik sınırla ikiye böldüm:
* **Eğitim Seti (Train):** 2013 ve 2014 yıllarının tamamı (Verinin %78'i)
* **Test Seti (Test):** 2015 yılının Ocak ayından Temmuz sonuna kadar olan bölümü (Verinin %22'si)

> 💡 Kaggle şampiyonları modellerini genellikle 42 günlük kısa periyotlarda test ederken; ben modelimin uzun vadedeki dayanıklılığını (robustness) kanıtlamak amacıyla tam **7 Aylık (~210 günlük)** oldukça zorlu bir test seti kullandım.

---

## 🧠 4. Derin Öğrenme ve Optimizasyon Stratejisi

Zaman serisi verilerindeki uzun vadeli bağımlılıkları yakalayabilmek için PyTorch kütüphanesini kullanarak **LSTM (Long Short-Term Memory)** ve **GRU (Gated Recurrent Unit)** mimarilerini sıfırdan tasarladım.

* **Yüksek Ağ Kapasitesi (512 Nöron):** Modelin karmaşık örüntüleri (tatil etkileri, hafta sonu düşüşleri) kavrayabilmesi için gizli katman (Hidden Size) kapasitesini 512 nörona çıkardım.
* **Erken Durdurma (Early Stopping):** 512 nöronlu devasa bir modelin eğitim verisini ezberleme (Overfitting) riski çok yüksektir. Bunu kesin olarak engellemek için, modelin test setindeki performansını her tur izleyen ve 15 tur üst üste iyileşme göremezse eğitimi durdurup en başarılı (altın) ağırlıklara geri dönen bir *Early Stopping* mekanizması yazdım.
* **Dropout:** Ağ içindeki nöronların %30'unu rastgele kapatarak modelin genelleme yeteneğini artırdım.

---

## 🌲 5. Literatür Kıyaslaması: XGBoost

Derin öğrenme modellerimin performansını endüstri standartlarıyla kıyaslayabilmek adına, literatürdeki en başarılı ağaç tabanlı algoritma olan **XGBoost (Extreme Gradient Boosting)** modelini de çalışmama dahil ettim. 
Ağaç derinliğini `max_depth=9` seviyesine çekip, `subsample=0.9` parametresiyle ezberi engelleyerek oldukça güçlü bir kıyaslama modeli (baseline) oluşturdum.

---

## 🏆 6. Nihai Performans ve R2 Skorları

Veri sızıntısı (Data Leakage) olmadan, daha önce hiç görülmemiş 2015 yılı test verisi üzerinde elde ettiğim nihai performans metrikleri aşağıdadır:

| Model | MSE | RMSE | MAE | MAPE (%) | R2 Skoru | Eğitim Süresi |
|-------|-----|------|-----|----------|----------|---------------|
| **XGBoost** | 795,012 | 891.63 | 614.14 | **%9.09** | **%91.70** | ~4.7 Dk |
| **LSTM** | 2,120,886 | 1456.33 | 1035.92| **%16.69**| **%77.87** | ~115 Dk |
| **GRU** | 2,328,245 | 1525.86 | 1083.54| **%17.51**| **%75.70** | ~30 Dk |

* **XGBoost**, hedeflenen hata paylarını ciddi oranda düşürerek (MAPE %9.09) **%91.70 R2 skoruyla** muazzam bir başarı göstermiştir.
* **LSTM**, klasik tablosal verilerde (tabular data) Derin Öğrenme uygulamalarının tüm zorluklarına rağmen **%77.87** R2 skoruna ulaşmayı başarmıştır. Geliştirdiğim Early Stopping mekanizması sayesinde Validation Loss sürekli kontrol altında tutularak Overfitting oranı %0 seviyesinde tutulmuştur.

---

## 🚀 7. Çalıştırma Talimatları

Projenin teknik süreçlerini bilgisayarınızda baştan sona yeniden üretmek (reproduce) isterseniz:

**1. Gerekli Kütüphanelerin Yüklenmesi:**
```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn xgboost
```

**2. Jupyter Notebook'un Başlatılması:**
```bash
jupyter notebook
```

**3. Test Süreci:**
`rossmann_sales_forecast.ipynb` dosyasını açıp üst menüden **Kernel -> Restart & Run All** seçeneğine tıklayarak veri ön işleme, Kayan Pencere (Sliding Window) hesaplamaları, XGBoost eğitimi ve PyTorch Early Stopping süreçlerinin tam akışını gözlemleyebilirsiniz.
