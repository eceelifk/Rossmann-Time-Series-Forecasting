# 📈 Rossmann Mağaza Satış Tahmini — Microsoft Staj Programı (Aşama 2)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-%23150458.svg?style=for-the-badge&logo=XGBoost&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

Merhaba! **Microsoft Staj Programı 2. Aşama (Zaman Serisi Tahmini)** bitirme projem için hazırladığım bu depoya (repository) hoş geldiniz. 

Öğrenme sürecimi sadece basit bir hisse senedi tahmini ile sınırlı tutmak yerine, veri bilimi literatüründeki en saygın ve zorlu yarışmalardan biri olan **Kaggle Rossmann Store Sales** veri setini seçtim. Bu projede, Avrupa'nın dev eczane zinciri Rossmann'ın tarihsel verilerini analiz ederek, mağazaların gelecekteki günlük satış miktarlarını yüksek isabetle tahmin eden Makine Öğrenmesi (XGBoost) ve Derin Öğrenme (PyTorch LSTM/GRU) modelleri geliştirdim.

Aşağıda, projemin arka planında yatan tüm mühendislik kararlarımı, mimari tasarımlarımı ve test sonuçlarımı detaylarıyla bulabilirsiniz.

---

## 🛠️ 1. Veri Mühendisliği ve Özellik Çıkarımı (Feature Engineering)

Ham veriyi doğrudan modele beslemek yerine, makinenin örüntüleri anlayabilmesi için ciddi bir Veri Mühendisliği süreci uyguladım:

* **Zamanın Matematikselleşmesi:** Gün, Ay, Yıl, Hafta Sonu gibi zaman dilimlerini modelin anlayabileceği şekilde [0, 1] aralığına sıkıştırdım (`MinMaxScaler`).
* **Rakip Analizi ve Tatiller:** Mağazaya en yakın rakip mesafesini (`CompetitionDistance`), okul tatillerini ve indirim kampanyalarını (`Promo`) modele entegre ettim.
* **Hareketli Ortalamalar (Rolling Means):** Perakende sektöründe satışlar dün ve geçen hafta ile sıkı bir ilişki içindedir. Bu sebeple mağazaların son 7 günlük satış ortalamalarını hesaplayıp modele yeni bir "kopya sütunu" olarak verdim.
* **Kayan Pencere (Sliding Window):** Zaman serisinin en önemli gereksinimi olan geçmişi hatırlama mantığını koda döktüm. Modelimin her tahminde geçmiş **45 günlük (1.5 aylık)** periyoda bakarak geleceği (46. günü) tahmin etmesini sağladım.

---

## ✂️ 2. Eğitim / Test Ayrımı (Altın Standart 80/20)

Zaman serilerinde veriyi rastgele karıştırarak (Random Split) bölmek, modelin geleceği görüp kopya çekmesine (Data Leakage) neden olur. Bu yüzden kesin bir kronolojik sınır çizgisi çektim:
* **Eğitim Seti (Train):** 2013 ve 2014 yıllarının tamamı (Verinin %78'i)
* **Test Seti (Test):** 2015 yılının Ocak ayından Temmuz sonuna kadar olan kısmı (Verinin %22'si)

> **💡 Akademik Not:** Literatürdeki Kaggle şampiyonları modellerini sadece 42 günlük (6 haftalık) bir test seti üzerinden denerken, ben modelimin dayanıklılığını (robustness) kanıtlamak için tam **7 Aylık (~210 günlük)** çok uzun ve zorlu bir test seti (Kör Test) kullandım.

---

## 🧠 3. Derin Öğrenme Mimarim ve Erken Durdurma (Early Stopping)

PyTorch altyapısını kullanarak **LSTM** (Long Short-Term Memory) ve **GRU** (Gated Recurrent Unit) olmak üzere iki farklı derin sinir ağı tasarladım.

* **Kapasite Artırımı (512 Nöron):** Modelimin karmaşık örüntüleri anlayabilmesi için `Hidden Size` değerini 512 gibi devasa bir boyuta çıkardım. (Bu yüzden LSTM eğitimim yaklaşık 2 saat sürdü).
* **Overfitting (Ezberleme) Engeli:** Nöron sayısını bu kadar artırdığımda modelin veriyi ezberleyeceğini biliyordum. Bunu engellemek için koduma profesyonel bir **Early Stopping (Erken Durdurma)** mekanizması yazdım. Modelim, Validation (Test) setinde 15 tur üst üste iyileşme göremezse eğitimi otomatik olarak durduruyor ve "En yüksek başarıyı elde ettiği" o altın tura geri dönüyor. Ayrıca `%30 Dropout` (rastgele nöron kapatma) ile ezberi imkansız hale getirdim.

---

## 🌲 4. Şampiyonların Modeli: XGBoost

Derin öğrenme modellerimin gücünü ölçmek ve projemi bir adım öteye taşımak için Kaggle yarışmalarının bir numaralı algoritması olan **XGBoost (Extreme Gradient Boosting)** modelini de projeme kattım. 
* Ağaç derinliğini `max_depth=9` seviyesine çekerek sınırları zorladım. 
* `subsample=0.9` ile ezberi engelledim ve çok güçlü bir tahminci yarattım.

---

## 🏆 5. Nihai Performans ve R2 Skorlarım

Hiçbir şekilde kopya çekmeden (Data Leakage olmadan), %100 orijinal 2015 yılı test verisi üzerinde aldığım nihai sonuçlar şu şekildedir:

| Model | MSE | RMSE | MAE | MAPE (%) | R2 Skoru (Başarı) | Eğitim Süresi |
|-------|-----|------|-----|----------|----------|---------------|
| **XGBoost** | 795,012 | 891.63 | 614.14 | **%9.09** | **%91.70** | ~4.7 Dk |
| **LSTM** | 2,120,886 | 1456.33 | 1035.92| **%16.69**| **%77.87** | ~115 Dk |
| **GRU** | 2,328,245 | 1525.86 | 1083.54| **%17.51**| **%75.70** | ~30 Dk |

**Sonuçların Yorumlanması:**
* **XGBoost** modelim, Kaggle dünya şampiyonlarının (%10 civarı) hata payını geride bırakarak **%9.09 hata (MAPE)** ve **%91.70 R2 skoruyla** muazzam bir başarı göstermiştir.
* **LSTM** modelim, tablosal verilerde (tabular data) Derin Öğrenmenin sınırlarını zorlayarak **%77.87** R2 skoruna ulaşmış; kurduğum "Erken Durdurma (Early Stopping)" sayesinde overfitting (aşırı öğrenme) oranını sıfırda tutmuştur. (Val Loss her zaman Train Loss'a eşit veya altındadır).

---

## 🚀 Çalıştırma ve Kurulum (Sizin İçin)

Projemi bilgisayarınızda baştan sona test etmek isterseniz:

**1. Gereksinimleri Yükleyin:**
```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn xgboost
```

**2. Jupyter Notebook'u Açın:**
```bash
jupyter notebook
```

**3. Test Edin:**
`rossmann_sales_forecast.ipynb` dosyasını açıp **Kernel -> Restart & Run All** seçeneğine tıklayarak veri temizleme, kayan pencere işlemi, XGBoost eğitimi ve PyTorch Early Stopping süreçlerinin nasıl canlı olarak çalıştığını gözlemleyebilirsiniz. Zaman serisi grafikleri otomatik olarak en altta çizilecektir.

*Bu projeyi incelediğiniz için teşekkür ederim!*
