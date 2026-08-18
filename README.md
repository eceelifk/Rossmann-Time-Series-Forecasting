# Rossmann Mağaza Satış Tahmini (Aşama 2 Projesi)

Bu projeyi, Microsoft Staj Programım kapsamındaki "Aşama 2 (Zaman Serisi Tahmin Modeli)" gereksinimlerine uygun olarak PyTorch kullanarak geliştirdim. 
Projemin amacı, Kaggle'daki Rossmann veri setini kullanarak geçmiş mağaza satışlarına dayanarak gelecekteki mağaza satışlarını (regresyon problemi) tahmin etmektir.

## Proje Hakkında
Projemi geliştirirken, derin öğrenme tabanlı zaman serisi mimarilerinden olan **LSTM (Long Short-Term Memory)** ve **GRU (Gated Recurrent Unit)** modellerini kullandım. 
Verileri "Kayan Pencere (Sliding Window)" yaklaşımı ile PyTorch tensörlerine uyarladım. Kapsamlı bir çalışma olması için, sadece tek bir mağazayı değil tüm mağazaların verilerini modele dahil ederek genel bir tahmin modeli eğittim.

## Kullanılan Teknolojiler
- **Python 3.x**
- **PyTorch** (Derin Öğrenme / Model Kurulumu ve Eğitimi)
- **Pandas** (Veri Analizi ve Ön İşleme)
- **Scikit-Learn** (MinMaxScaler ile Ölçeklendirme ve MSE/RMSE Hesaplamaları)
- **Matplotlib** (Elde Ettiğim Sonuçların Görselleştirilmesi)

## Dosya Yapısı
- `rossmann_sales_forecast.ipynb`: Veri ön işleme, veri ölçeklendirme, model oluşturma (LSTM ve GRU), eğitim ve test aşamalarının tamamını kodladığım ana Jupyter Notebook dosyasıdır.
- `train.csv` / `test.csv`: Kaggle Rossmann Store Sales yarışmasından kullandığım ham veri setidir (Kodların çalışması için bu dizinde yer alması gereklidir).

## Modeller ve Karşılaştırma
Jupyter Notebook'u çalıştırdığınızda, kurduğum her iki modelin (LSTM ve GRU) test seti üzerindeki Mean Squared Error (MSE) ve Root Mean Squared Error (RMSE) değerlerini karşılaştırmalı olarak görebilirsiniz. Ayrıca modellerin ne kadar sürede eğitildiğini de saniye cinsinden ölçtüm. 

Son adımda, modellerimin tahmin ettiği satış değerleri ile gerçek satış değerlerini Matplotlib kullanarak çizgi grafiği (Line Plot) üzerinde görselleştirdim.

## Kurulum ve Kullanım
Projemi kendi bilgisayarınızda (veya Anaconda ortamında) çalıştırmak için şu adımları izleyebilirsiniz:

1. Gerekli Python kütüphanelerinin sisteminizde kurulu olduğundan emin olun. Değilse terminalden (veya Anaconda Prompt üzerinden) indirebilirsiniz:
   ```bash
   pip install pandas numpy torch scikit-learn matplotlib
   ```
2. Proje dizinine gidip terminalden jupyter ortamını başlatın:
   ```bash
   jupyter notebook
   ```
3. `rossmann_sales_forecast.ipynb` dosyasını açıp **"Run All"** komutuyla veya hücreleri teker teker çalıştırarak projemin tamamını test edebilir ve çıktıları inceleyebilirsiniz.

