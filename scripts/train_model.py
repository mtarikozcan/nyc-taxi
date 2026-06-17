"""
NYC Taksi Talep Tahmini - Model Eğitim Scripti
================================================
Ankara Üniversitesi - 3522 Bulut Bilişim Dersi
Hazırlayan: Mehmet Tarik Ozcan (222904436)

Bu script:
1. NYC TLC'den Ocak 2023 sarı taksi verisini indirir
2. Veri temizleme ve filtreleme uygular
3. Öznitelik mühendisliği yapar
4. 4 farklı model eğitir ve karşılaştırır
5. En iyi modeli (XGBoost) ve zone istatistiklerini kaydeder
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings('ignore')

# Çıktı dizini (api/ klasörü)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
API_DIR = os.path.join(PROJECT_DIR, "api")
MODEL_PATH = os.path.join(API_DIR, "taxi_demand_model.pkl")
ZONE_STATS_PATH = os.path.join(API_DIR, "zone_stats.json")

# Veri URL
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"


def print_header(text):
    """Bölüm başlığı yazdır."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_step(text):
    """Adım bilgisi yazdır."""
    print(f"  -> {text}")


# ============================================================
# 1. VERİ İNDİRME
# ============================================================
def load_data():
    print_header("1. VERI INDIRME")
    print_step(f"Kaynak: {DATA_URL}")
    print_step("NYC TLC Ocak 2023 sari taksi verisi indiriliyor...")
    print_step("(~600 MB, internet hizina bagli olarak birkac dakika surebilir)")

    # Yerel dosyaya indir (retry mekanizmasi ile)
    local_file = os.path.join(SCRIPT_DIR, "yellow_tripdata_2023-01.parquet")

    if os.path.exists(local_file) and os.path.getsize(local_file) > 100_000_000:
        print_step(f"Yerel dosya bulundu: {local_file}")
        print_step("Tekrar indirmek yerine yerel dosya kullaniliyor...")
    else:
        import urllib.request
        import ssl

        # SSL context - baglanti sorunlarini asma
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                print_step(f"Indirme denemesi {attempt}/{max_retries}...")
                urllib.request.urlretrieve(DATA_URL, local_file)
                print_step("Indirme tamamlandi!")
                break
            except Exception as e:
                print_step(f"Deneme {attempt} basarisiz: {e}")
                if attempt == max_retries:
                    print_step("Tum denemeler basarisiz. Alternatif yontem deneniyor...")
                    # requests kutuphanesi ile dene
                    try:
                        import subprocess
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", "requests"
                        ], capture_output=True)
                        import requests as req_lib
                        print_step("requests ile indiriliyor...")
                        r = req_lib.get(DATA_URL, stream=True, verify=False, timeout=120)
                        r.raise_for_status()
                        with open(local_file, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192*1024):
                                f.write(chunk)
                        print_step("Indirme tamamlandi (requests ile)!")
                    except Exception as e2:
                        raise RuntimeError(
                            f"Veri indirilemedi. Hata: {e2}\n"
                            f"Manuel indirme: {DATA_URL}\n"
                            f"Dosyayi buraya kaydedin: {local_file}"
                        )
                else:
                    wait = attempt * 5
                    print_step(f"{wait} saniye bekleniyor...")
                    time.sleep(wait)

    start = time.time()
    df = pd.read_parquet(local_file)
    elapsed = time.time() - start

    print_step(f"Dosya okundu! ({elapsed:.1f} saniye)")
    print_step(f"Ham veri boyutu: {df.shape[0]:,} satir x {df.shape[1]} sutun")
    print_step(f"Bellek kullanimi: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return df


# ============================================================
# 2. VERİ TEMİZLEME (Doküman Bölüm 3)
# ============================================================
def clean_data(df):
    print_header("2. VERİ TEMİZLEME")
    original_count = len(df)
    print_step(f"Temizlik oncesi: {original_count:,} satir")

    # Tarih aralığı filtresi
    df = df[(df['tpep_pickup_datetime'] >= '2023-01-01') &
            (df['tpep_pickup_datetime'] < '2023-02-01')]
    print_step(f"Tarih filtresi sonrasi: {len(df):,} satir")

    # Mesafe filtresi
    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]
    print_step(f"Mesafe filtresi sonrasi: {len(df):,} satir")

    # Ücret filtresi
    df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 500)]
    print_step(f"Ucret filtresi sonrasi: {len(df):,} satir")

    # Yolcu filtresi
    df = df[df['passenger_count'] > 0]
    print_step(f"Yolcu filtresi sonrasi: {len(df):,} satir")

    # Bölge filtresi
    df = df[(df['PULocationID'] > 0) & (df['PULocationID'] < 264)]
    print_step(f"Bolge filtresi sonrasi: {len(df):,} satir")

    removed = original_count - len(df)
    pct = (removed / original_count) * 100
    print_step(f"Cikarilan: {removed:,} satir ({pct:.1f}%)")
    print_step(f"Temizlik sonrasi: {len(df):,} satir [OK]")

    return df


# ============================================================
# 3. ÖZNİTELİK MÜHENDİSLİĞİ (Doküman Bölüm 5)
# ============================================================
def engineer_features(df):
    print_header("3. ÖZNİTELİK MÜHENDİSLİĞİ")

    # Zamansal oznitelikler turetme
    print_step("Zamansal oznitelikler turetiliyor...")
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['day'] = df['tpep_pickup_datetime'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Talep agregasyonu: bolge + gun + saat bazinda gruplama
    print_step("Bolge + gun + saat bazinda gruplama yapiliyor...")
    demand_df = df.groupby(['PULocationID', 'day', 'hour']).agg(
        trip_count=('VendorID', 'count'),
        avg_distance=('trip_distance', 'mean'),
        avg_fare=('fare_amount', 'mean'),
        avg_passengers=('passenger_count', 'mean')
    ).reset_index()

    # day_of_week ve is_weekend türetmek için orijinal tarih bilgisini kullan
    # day sütununu kullanarak day_of_week hesapla
    day_dow_map = df.groupby('day')['day_of_week'].first().to_dict()
    demand_df['day_of_week'] = demand_df['day'].map(day_dow_map)
    demand_df['is_weekend'] = (demand_df['day_of_week'] >= 5).astype(int)

    print_step(f"Toplam {len(demand_df):,} bolge-gun-saat kombinasyonu olusturuldu")

    # Final öznitelik matrisi
    feature_cols = ['PULocationID', 'hour', 'day_of_week', 'is_weekend',
                    'avg_distance', 'avg_fare', 'avg_passengers']
    target_col = 'trip_count'


    print_step(f"Final oznitelik sayisi: {len(feature_cols)}")
    for i, col in enumerate(feature_cols, 1):
        print_step(f"  {i}. {col}: [{demand_df[col].min():.2f} - {demand_df[col].max():.2f}]")

    print_step(f"Hedef degisken (trip_count): ortalama={demand_df[target_col].mean():.1f}, "
               f"max={demand_df[target_col].max()}")

    return demand_df, feature_cols, target_col


# ============================================================
# 4. ZONE İSTATİSTİKLERİ OLUŞTURMA
# ============================================================
def create_zone_stats(demand_df):
    print_header("4. ZONE İSTATİSTİKLERİ")

    zone_stats = demand_df.groupby('PULocationID').agg(
        avg_distance=('avg_distance', 'mean'),
        avg_fare=('avg_fare', 'mean'),
        avg_passengers=('avg_passengers', 'mean'),
        avg_trips=('trip_count', 'mean'),
        total_trips=('trip_count', 'sum')
    ).reset_index()

    print_step(f"Toplam {len(zone_stats)} aktif bolge tespit edildi")

    # JSON formatına dönüştür
    zone_dict = {}
    for _, row in zone_stats.iterrows():
        zone_id = int(row['PULocationID'])
        zone_dict[str(zone_id)] = {
            "zone_id": zone_id,
            "avg_distance": round(float(row['avg_distance']), 4),
            "avg_fare": round(float(row['avg_fare']), 4),
            "avg_passengers": round(float(row['avg_passengers']), 4),
            "avg_trips": round(float(row['avg_trips']), 2),
            "total_trips": int(row['total_trips'])
        }

    # zone_stats.json kaydet
    with open(ZONE_STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump(zone_dict, f, indent=2)

    print_step(f"zone_stats.json kaydedildi: {ZONE_STATS_PATH}")
    print_step(f"En yogun bolge: Zone {max(zone_dict.values(), key=lambda x: x['total_trips'])['zone_id']}")

    return zone_dict


# ============================================================
# 5. MODEL EĞİTİMİ (Doküman Bölüm 6)
# ============================================================
def train_models(demand_df, feature_cols, target_col):
    print_header("5. MODEL EĞİTİMİ")

    X = demand_df[feature_cols].values
    y = demand_df[target_col].values

    # %80/%20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print_step(f"Egitim seti: {X_train.shape[0]:,} satir")
    print_step(f"Test seti: {X_test.shape[0]:,} satir")

    # Model tanımları (doküman hiperparametreleri)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=15,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=6,
            learning_rate=0.1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=8,
            learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            verbosity=0
        )
    }

    results = {}
    best_model = None
    best_r2 = -float('inf')
    best_name = ""

    for name, model in models.items():
        print(f"\n  [*] {name} egitiliyor...")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 4),
            "train_time": round(train_time, 2)
        }

        marker = " << EN IYI!" if r2 > best_r2 else ""
        print(f"     RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f} | "
              f"Sure: {train_time:.2f}s{marker}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    return results, best_model, best_name, best_r2


# ============================================================
# 6. MODEL KARŞILAŞTIRMA VE KAYDETME (Doküman Bölüm 7)
# ============================================================
def save_results(results, best_model, best_name, best_r2):
    print_header("6. SONUÇLAR VE KAYDETME")

    # Karşılaştırma tablosu
    print("\n  ┌─────────────────────┬─────────┬────────┬────────┬──────────┐")
    print("  │ Model               │  RMSE ↓ │ MAE ↓  │  R² ↑  │ Süre (s) │")
    print("  ├─────────────────────┼─────────┼────────┼────────┼──────────┤")
    for name, r in results.items():
        marker = " ⭐" if name == best_name else "  "
        print(f"  │{marker}{name:<19}│ {r['rmse']:>7} │ {r['mae']:>6} │ {r['r2']:>6} │ {r['train_time']:>8} │")
    print("  └─────────────────────┴─────────┴────────┴────────┴──────────┘")

    # En iyi modeli kaydet
    print(f"\n  [KAZANAN] Model: {best_name} (R2 = {best_r2:.4f})")
    joblib.dump(best_model, MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
    print_step(f"Model kaydedildi: {MODEL_PATH} ({model_size:.2f} MB)")



# ============================================================
# ANA FONKSİYON
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  NYC TAKSI TALEP TAHMINI - MODEL EGITIM SCRIPTI")
    print("  Ankara Üniversitesi - 3522 Bulut Bilişim Dersi")
    print("=" * 60)

    total_start = time.time()

    # 1. Veri yükleme
    df = load_data()

    # 2. Veri temizleme
    df = clean_data(df)

    # 3. Öznitelik mühendisliği
    demand_df, feature_cols, target_col = engineer_features(df)

    # 4. Zone istatistikleri oluştur
    create_zone_stats(demand_df)

    # Bellek temizleme
    del df

    # 5. Model eğitimi
    results, best_model, best_name, best_r2 = train_models(
        demand_df, feature_cols, target_col
    )

    # 6. Sonuçları kaydet
    save_results(results, best_model, best_name, best_r2)

    total_time = time.time() - total_start
    print_header("TAMAMLANDI")
    print_step(f"Toplam sure: {total_time:.1f} saniye")
    print_step(f"Model: {MODEL_PATH}")
    print_step(f"Zone stats: {ZONE_STATS_PATH}")
    print_step("Sonraki adim: cd api && uvicorn main:app --port 8000 --reload")
    print()


if __name__ == "__main__":
    main()
