# Rossmann Mağaza Satış Tahmini: Proje Gelişim ve Optimizasyon Raporu

**Son Güncelleme Tarihi:** 2026-08-24
**Revizyon (Değişim) Sayısı:** 7. Sürüm (Nihai Revizyon)

Bu doküman, zaman serisi satış tahmini projesinin başlangıç aşamasından nihai teslim aşamasına kadar geçirdiği evrimleri, uygulanan makine öğrenmesi tekniklerini ve bu adımların modelin R2 skoru ile hata payı (MAPE) üzerindeki kanıtlanmış etkilerini adım adım özetlemektedir.

### 🔴 1. Aşama: Temel (Baseline) Modelin Kurulması
Projenin ilk aşamasında sadece geçmiş satış verileri kullanılarak temel LSTM ve GRU modelleri kuruldu. Test ve Eğitim ayrımı rastgele %80-%20 oranında yapıldı.
* **Eğitim (Epoch):** 2 Tur.
* **Zaman Penceresi:** 7 Gün.
* **Durum:** Model, promosyonları ve tatilleri bilmediği için sıradan bir başarı gösterdi. Ayrıca mağazaların kapalı olduğu (Satış=0) Pazar günleri, standart MAPE formülünde Sıfıra Bölme hatası yarattığı için hata payı hesabı bozuldu.
* **R2 Skoru:** ~%64 (0.64)

### 🟡 2. Aşama: Öznitelik Mühendisliği ve Overfitting Koruması
* **Feature Engineering:** Promo, SchoolHoliday ve DayOfWeek verileri eklendi.
* **Regularization:** %20 Dropout ve Weight Decay (L2) uygulandı. Batch Size 1024 yapıldı.
* **Sıfıra Bölme Çözümü:** `safe_mape` fonksiyonu yazılarak sıfır olan günler formülden maskelendi.
* **Sonuç:** R2 Skoru ~%74.5, Hata Payı (MAPE) ise %17.6 civarına geriledi.

### 🟠 3. Aşama: Bilimsel Test Doğrulaması (Yıllara Göre Bölme)
* **Yeni Train/Test Ayrımı:** 2013 ve 2014 yıllarının TAMAMI modele öğretildi. Test seti olarak 2015 yılı ayrıldı.
* **Sonuç:** Model bu çok ağır test koşullarında dahi çökmedi ve R2 başarısını %74.32 bandında korudu.

### 🟢 4. Aşama: Veri Zenginleştirmesi (Merge Store.csv)
`store.csv` dosyası modele entegre edilerek mağazaların genetik yapıları makineye öğretildi.
* **One-Hot Encoding:** Mağaza Tipi ve Ürün Çeşitliliği kodlandı. Rakip Mağaza Uzaklığı eklendi.
* **Sonuç:** Özellik sayısı 13'e fırladı. R2 Skoru %74.89'a yükseldi.

### 🚀 5. Aşama: Kaggle Optimizasyonları (Rolling Means ve Seed)
* **Geçmişin Ortalamaları (Rolling Means):** Modele "Son 7 Günün Satış Ortalaması" bilgisi eklendi.
* **Genişletilmiş Feature Engineering:** Veri setinden Ay (Month), Gün (Day) ve StateHoliday (Resmi Tatil) günleri modele entegre edildi.
* **Sonuç:** LSTM Skoru %76.92'ye yükseldi.

---

### 🏆 6. Aşama: XGBoost Entegrasyonu
Projenin sürümünde, ağaç tabanlı XGBoost algoritması 3. model olarak dahil edildi.
* **XGBoost Rekoru:** 3 boyutlu zaman serisi verisi düz tabloya çevrilerek GPU destekli XGBoost'a verildi. XGBoost sadece 9 saniyede inanılmaz bir doğruluk oranına ulaştı.

---

### 🔥 7. Aşama (Güncel Final Durumu): Epoch ve Hidden Size Optimizasyonu
Derin Öğrenme modellerinin (LSTM ve GRU) kapasitesini artırmak ve XGBoost'a yaklaşmalarını sağlamak amacıyla mimari güçlendirildi.

* **Epoch Artışı:** Derin öğrenme modelleri (LSTM ve GRU) 10 yerine **40 Epoch** boyunca eğitildi.
* **Hidden Size Artışı:** Modellerin gizli katman boyutu 128'den **256'ya** çıkarıldı. Böylece modellerin işlem kapasitesi 4 katına ulaştı.

#### Sonuç Tablosu (2015 Test Seti Üzerinden Nihai Değerler)

| Model | MSE | RMSE | MAE | MAPE (Hata Payı) | R2 Skoru (Başarı) | Eğitim Süresi |
|-------|-----|------|-----|------------------|-------------------|---------------|
| **LSTM** | 2171913 | 1473.74 | 1050.76 | %17.37 | **%76.92** | 543.4 sn |
| **GRU**  | 2298115 | 1515.95 | 1079.60 | %17.86 | **%75.58** | 458.7 sn |
| **XGBoost**| 999776 | 999.89 | 687.44 | %10.43 | **%91.08** | 8.7 sn |

**🎯 Nihai Değerlendirme:** 
Son optimizasyonlarla birlikte (Epoch=40, Hidden_Size=256) derin öğrenme modellerimiz olan LSTM ve GRU, artırılmış kapasiteleri ile %75-%77 başarı bandına güçlü bir şekilde oturmuştur. Kaggle şampiyonlarının tercihi olan **XGBoost** modeli ise %91.08 gibi kusursuz bir başarıya imza atarak tabüler verideki üstünlüğünü korumuştur. Grafik incelendiğinde XGBoost'un ani tatil satışlarını ve sıfıra düşen günleri çok net yakaladığı; LSTM ve GRU'nun ise genel trendleri başarıyla kavradığı görülmektedir. Proje, Makine Öğrenmesi ile Derin Öğrenmenin yeteneklerini kıyaslayan eksiksiz bir çalışma olmuştur.
