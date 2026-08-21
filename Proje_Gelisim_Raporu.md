# Rossmann Mağaza Satış Tahmini: Proje Gelişim ve Optimizasyon Raporu

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
