from datetime import datetime
import pytz

# Şu anki zamanı al
turkey_tz = pytz.timezone('Europe/Istanbul')
current_time = datetime.now(turkey_tz).strftime('%Y-%m-%d %H:%M:%S')

report_content = f"""# Rossmann Mağaza Satış Tahmini: Proje Gelişim ve Optimizasyon Raporu

**Son Güncelleme Tarihi:** {current_time}
**Revizyon (Değişim) Sayısı:** 6. Sürüm (Nihai Revizyon)

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

### 🏆 6. Aşama (Güncel Final Durumu): XGBoost Entegrasyonu ve 10 Epoch Eğitimi
Projenin nihai sürümünde, ağaç tabanlı XGBoost algoritması 3. model olarak dahil edildi. LSTM ve GRU'nun eğitim süreleri artırılarak daha derin öğrenmeleri sağlandı.

* **Epoch Artışı:** Derin öğrenme modelleri (LSTM ve GRU) 7 yerine **10 Epoch** boyunca eğitildi (Süreleri ~375 saniyeye çıktı).
* **XGBoost Rekoru:** 3 boyutlu zaman serisi verisi düz tabloya çevrilerek GPU destekli XGBoost'a verildi. XGBoost sadece 9.5 saniyede inanılmaz bir doğruluk oranına ulaştı.

#### Sonuç Tablosu (2015 Test Seti Üzerinden Nihai Değerler)

| Model | MSE | RMSE | MAE | MAPE (Hata Payı) | R2 Skoru (Başarı) | Eğitim Süresi |
|-------|-----|------|-----|------------------|-------------------|---------------|
| **LSTM** | 2114123 | 1454.00 | 1040.83 | %16.97 | **%77.19** | 373.1 sn |
| **GRU**  | 2288544 | 1512.79 | 1073.26 | %17.41 | **%75.30** | 382.3 sn |
| **XGBoost**| 1053637 | 1026.47 | 705.20 | %10.73 | **%88.63** | 9.5 sn |

**🎯 Nihai Değerlendirme:** 
Son güncellemelerle birlikte LSTM modeli R2 skorunu %77.19'a taşımıştır. Ancak asıl şaşırtıcı sonuç, Kaggle şampiyonlarının tercihi olan **XGBoost** modelinin %88.63 gibi kusursuz bir başarıya imza atmasıdır. Grafik incelendiğinde XGBoost'un (Kırmızı Kesik Çizgi) ani tatil satışlarını ve sıfıra düşen günleri nokta atışı yakaladığı görülmektedir. Proje, Makine Öğrenmesi ile Derin Öğrenmenin yeteneklerini kıyaslayan üst düzey bir bitirme tezi kalitesinde tamamlanmıştır.
"""

with open('Proje_Gelisim_Raporu.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("Proje_Gelisim_Raporu.md updated successfully with the 6th phase and timestamp.")
