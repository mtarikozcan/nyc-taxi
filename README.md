# NYC Taksi Talep Tahmini 🚕📊

Bu proje, New York City (NYC) sarı taksi verilerini kullanarak bölge, gün ve saat bazlı taksi talebini tahmin eden uçtan uca bir Makine Öğrenmesi (Machine Learning) web uygulamasıdır. Ankara Üniversitesi Bilgisayar Mühendisliği 3522 Bulut Bilişim Dersi için geliştirilmiştir.

![NYC Taxi Prediction](https://raw.githubusercontent.com/mtarikozcan/nyc-taxi/main/web/public/favicon.ico) *(Eğer önizleme görseli eklerseniz buraya koyabilirsiniz)*

## 🌟 Özellikler

- **Yüksek Doğruluklu Makine Öğrenmesi**: XGBoost modeli ile eğitilmiş sistem, taksi talebini `%93.7 (R²)` doğrulukla tahmin eder.
- **Etkileşimli Harita (Leaflet.js)**: NYC taksi bölgelerini ısı haritası (heatmap) şeklinde görselleştirir. Tıklanan bölgenin detaylarını ve tahmin sonucunu harita üzerinde dinamik olarak gösterir.
- **Modern ve Şık Arayüz**: Karanlık tema, cam efekti (glassmorphism) ve pürüzsüz mikro-animasyonlar ile desteklenmiş duyarlı (responsive) tasarım.
- **Hızlı ve Güvenilir API**: FastAPI ile desteklenen Python tabanlı tahmin motoru ve Express.js tabanlı Node web sunucusu.
- **Otomatik Veri Çekimi**: NYC TLC Shapefile verilerini ve büyük boyutlu Parquet dosyalarını otomatik olarak indirecek güvenli Python scriptleri.

## 🏗️ Mimari

Proje 3 temel bileşenden oluşmaktadır:

1. **Makine Öğrenmesi (Python Scripts)**:
   - NYC TLC Trip Record Data (Ocak 2023) kullanılarak veri ön işleme, özellik mühendisliği (feature engineering) ve model eğitimi yapılır.
   - En iyi performans gösteren **XGBoost** modeli `.joblib` formatında kaydedilir.
2. **Backend API (FastAPI)**:
   - Eğitilmiş modeli yükleyerek `/api/predict` ve `/api/zones` uç noktalarını (endpoints) sunar. Web arayüzünden gelen talepleri milisaniyeler içinde yanıtlar.
3. **Frontend & Web Server (Node.js + Express.js)**:
   - Statik HTML/CSS/JS dosyalarını sunar ve frontend ile FastAPI backend'i arasında proxy görevi görür. (CORS sorunlarını ortadan kaldırır).

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

### Ön Koşullar
- Node.js (v18+)
- Python (v3.10+)

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/mtarikozcan/nyc-taxi.git
cd taxi-demand-prediction
```

### 2. Modelin Eğitilmesi ve Harita Verilerinin İndirilmesi
Veri setini indirip modeli eğitmek ve harita (GeoJSON) verisini oluşturmak için:
```bash
cd scripts
pip install -r requirements-train.txt

# Modeli eğit
python train_model.py

# Harita bölge verilerini (GeoJSON) indir ve dönüştür
python download_geojson.py
cd ..
```
*Not: Veri indirme işlemleri dosya boyutlarına göre birkaç dakika sürebilir.*

### 3. API Sunucusunun Başlatılması
FastAPI sunucusunu başlatmak için yeni bir terminal açın:
```bash
cd api
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Web Sunucusunun Başlatılması
Farklı bir terminal açın ve Node.js sunucusunu başlatın:
```bash
cd web
npm install
node server.js
```

### 5. Uygulamayı Kullanın
Tarayıcınızı açın ve aşağıdaki adrese gidin:
**👉 http://localhost:3000**

## 📂 Proje Yapısı

```text
taxi-demand-prediction/
│
├── api/                   # FastAPI Backend
│   ├── main.py            # API uç noktaları
│   └── requirements.txt   # Backend bağımlılıkları
│
├── models/                # Eğitilmiş Model Dosyaları
│   ├── best_model.joblib  # XGBoost Modeli
│   └── zone_stats.json    # Bölge bazlı ortalama istatistikler
│
├── scripts/               # Veri Hazırlık ve Eğitim
│   ├── train_model.py     # Veri çekme ve model eğitim kodu
│   ├── download_geojson.py# Harita verisi (Shapefile -> GeoJSON)
│   └── requirements-train.txt
│
├── web/                   # Frontend & Node Sunucusu
│   ├── public/            # Statik Dosyalar (HTML, CSS, JS)
│   │   ├── index.html     # Ana arayüz ve Leaflet harita
│   │   └── nyc_taxi_zones.geojson # Harita sınır verileri
│   ├── package.json       # Node.js yapılandırması
│   └── server.js          # Express.js sunucu kodları
│
└── README.md              # Proje dokümantasyonu
```

## 👨‍💻 Hazırlayan
**Mehmet Tarik Ozcan**
Öğrenci No: 222904436
Ankara Üniversitesi · Bilgisayar Mühendisliği
3522 Bulut Bilişim Dersi Kapsamında Geliştirilmiştir.

## 📄 Lisans
Bu proje MIT Lisansı ile lisanslanmıştır. Veriler [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) üzerinden halka açık olarak sağlanmaktadır.
