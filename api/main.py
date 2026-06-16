"""
NYC Taksi Talep Tahmini - FastAPI Servisi
==========================================
Ankara Üniversitesi - 3522 Bulut Bilişim Dersi

Endpoint'ler:
  POST /predict  - Taksi talep tahmini
  GET  /zones    - Mevcut bölgeleri listele
  GET  /health   - Sağlık kontrolü
"""

import os
import json
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# Uygulama Yapılandırması
# ============================================================
app = FastAPI(
    title="NYC Taksi Talep Tahmini API",
    description="XGBoost modeli ile bölge ve zaman bazlı taksi talebi tahmini",
    version="1.0.0"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Model ve Zone Stats Yükleme
# ============================================================
API_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(API_DIR, "taxi_demand_model.pkl")
ZONE_STATS_PATH = os.path.join(API_DIR, "zone_stats.json")

# Model yükle
model = None
zone_stats = None
model_r2_score = 0.937  # Varsayılan R² skoru

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model yüklendi: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Model yüklenemedi: {e}")

try:
    with open(ZONE_STATS_PATH, 'r', encoding='utf-8') as f:
        zone_stats = json.load(f)
    print(f"✅ Zone stats yüklendi: {len(zone_stats)} bölge")
except Exception as e:
    print(f"⚠️ Zone stats yüklenemedi: {e}")


# ============================================================
# Request/Response Modelleri
# ============================================================
class PredictRequest(BaseModel):
    """Tahmin isteği modeli."""
    zone_id: int = Field(..., ge=1, le=263, description="PULocationID (1-263)")
    hour: int = Field(..., ge=0, le=23, description="Saat (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Haftanın günü (0=Pzt, 6=Paz)")


class PredictResponse(BaseModel):
    """Tahmin yanıtı modeli."""
    predicted_trips: int
    zone_id: int
    hour: int
    day_of_week: int
    is_weekend: int
    model: str = "XGBoost"
    r2_score: float


class HealthResponse(BaseModel):
    """Sağlık kontrolü yanıtı."""
    status: str
    model_loaded: bool
    zones_loaded: bool
    total_zones: int


# ============================================================
# Endpoint'ler
# ============================================================

@app.post("/predict", response_model=PredictResponse,
          summary="Taksi talep tahmini",
          description="Belirtilen bölge, saat ve gün için tahmin edilen trip sayısını döndürür.")
async def predict(request: PredictRequest):
    """
    Taksi talep tahmini yapar.
    
    Zone lookup mekanizması:
    - Kullanıcı sadece zone_id, hour, day_of_week sağlar
    - avg_distance, avg_fare, avg_passengers → zone_stats.json'dan doldurulur
    - is_weekend → day_of_week'ten otomatik türetilir
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model henüz yüklenmedi. Lütfen önce train_model.py scriptini çalıştırın."
        )

    if zone_stats is None:
        raise HTTPException(
            status_code=503,
            detail="Zone istatistikleri yüklenmedi. Lütfen önce train_model.py scriptini çalıştırın."
        )

    # Zone ID kontrolü
    zone_key = str(request.zone_id)
    if zone_key not in zone_stats:
        available_zones = sorted([int(k) for k in zone_stats.keys()])
        raise HTTPException(
            status_code=400,
            detail=f"Zone {request.zone_id} bulunamadı. "
                   f"Mevcut bölge sayısı: {len(available_zones)}. "
                   f"Örnek bölgeler: {available_zones[:10]}..."
        )

    # Zone lookup: eksik öznitelikleri zone_stats'tan doldur
    zone = zone_stats[zone_key]
    is_weekend = 1 if request.day_of_week >= 5 else 0

    # 7 öznitelikli vektör oluştur (doküman sırası ile)
    features = np.array([[
        request.zone_id,        # PULocationID
        request.hour,           # hour
        request.day_of_week,    # day_of_week
        is_weekend,             # is_weekend
        zone["avg_distance"],   # avg_distance
        zone["avg_fare"],       # avg_fare
        zone["avg_passengers"]  # avg_passengers
    ]])

    # Tahmin
    prediction = model.predict(features)
    predicted_trips = max(0, int(round(prediction[0])))

    return PredictResponse(
        predicted_trips=predicted_trips,
        zone_id=request.zone_id,
        hour=request.hour,
        day_of_week=request.day_of_week,
        is_weekend=is_weekend,
        model="XGBoost",
        r2_score=model_r2_score
    )


@app.get("/zones",
         summary="Mevcut bölgeleri listele",
         description="Tahmin yapılabilecek bölgelerin ID, ortalama mesafe, ücret ve yolcu istatistiklerini döndürür.")
async def get_zones():
    """Tüm aktif bölgelerin istatistiklerini döndürür."""
    if zone_stats is None:
        raise HTTPException(
            status_code=503,
            detail="Zone istatistikleri yüklenmedi."
        )

    zones_list = sorted(zone_stats.values(), key=lambda x: x["total_trips"], reverse=True)
    return {
        "total_zones": len(zones_list),
        "zones": zones_list
    }


@app.get("/health", response_model=HealthResponse,
         summary="Sağlık kontrolü",
         description="API'nin çalışır durumda olduğunu ve modelin yüklü olup olmadığını kontrol eder.")
async def health_check():
    """API sağlık durumunu kontrol eder."""
    total_zones = len(zone_stats) if zone_stats else 0
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        zones_loaded=zone_stats is not None,
        total_zones=total_zones
    )


# ============================================================
# Ana Giriş Noktası
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
