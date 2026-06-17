# Cloud Run Deploy Rehberi

Bu proje iki Cloud Run servisi olarak deploy edilir:

- **taxi-api** — FastAPI backend (XGBoost modeli ile tahmin)
- **taxi-web** — Node.js / Express frontend ve `/api/*` proxy

`taxi-web`, gelen `/api/*` isteklerini `API_TARGET` ortam değişkeninde tanımlı `taxi-api` Cloud Run URL'sine yönlendirir.

## Ön Koşullar

### Gerekli GCP API'leri
Aşağıdaki API'lerin GCP projesinde etkin olması gerekir:

- `run.googleapis.com` (Cloud Run)
- `cloudbuild.googleapis.com` (Cloud Build — Dockerfile'dan build)
- `artifactregistry.googleapis.com` (Build çıktılarının depolanması)

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### GitHub Bağlantısı
Cloud Run konsolundan **"Connect Repository"** akışı ile GitHub hesabınızı bağlayın ve `taxi-demand-prediction` reposunu seçin. Branch olarak `main` (veya deploy ettiğiniz branch) seçilir.

## 1) `taxi-api` Servisi (FastAPI)

| Ayar | Değer |
|------|-------|
| Service name | `taxi-api` |
| Region | `europe-west1` (veya tercih) |
| Source | GitHub repo, branch `main` |
| Build type | **Dockerfile** |
| Source location | `/api` |
| Dockerfile path | `api/Dockerfile` |
| Port | `8080` (container port) |
| CPU | 1 |
| Memory | 1 GiB (XGBoost + pandas için yeterli) |
| Authentication | **Allow unauthenticated** |
| Min instances | 0 |
| Max instances | 3 |

### Environment Variables
- `PORT` → Cloud Run otomatik set eder, elle değiştirmeyin.

Diğer env gerekmez. Model dosyası (`taxi_demand_model.pkl`, 4.4 MB) ve `zone_stats.json` imaja gömülüdür.

Deploy bittikten sonra `taxi-api` Cloud Run URL'sini not alın (örn. `https://taxi-api-xxxxx-ew.a.run.app`).

## 2) `taxi-web` Servisi (Node/Express)

| Ayar | Değer |
|------|-------|
| Service name | `taxi-web` |
| Region | `europe-west1` (api ile aynı) |
| Source | GitHub repo, branch `main` |
| Build type | **Dockerfile** |
| Source location | `/web` |
| Dockerfile path | `web/Dockerfile` |
| Port | `8080` |
| CPU | 1 |
| Memory | 512 MiB |
| Authentication | **Allow unauthenticated** |
| Min instances | 0 |
| Max instances | 3 |

### Environment Variables
- `API_TARGET` → `https://taxi-api-xxxxx-ew.a.run.app` (yukarıdaki `taxi-api` URL'si — `/api` veya `/` ile bitirmeyin)
- `PORT` → Cloud Run otomatik set eder.

Deploy bittikten sonra `taxi-web` URL'si uygulamanın ana adresidir.

## Hızlı Doğrulama
```bash
curl https://taxi-api-xxxxx-ew.a.run.app/health
# {"status":"healthy","model_loaded":true,"zones_loaded":true,"total_zones":...}

curl https://taxi-web-xxxxx-ew.a.run.app/api/health
# Aynı yanıt (proxy üzerinden)
```

## Veri & Model Notları
- **Veritabanı kullanılmıyor.** Tüm tahmin tek bir `.pkl` modeli ve `zone_stats.json` ile yapılır.
- Model + GeoJSON + zone stats toplam ~9 MB; doğrudan imaja gömülüdür. GCS gerekmez.
- Cloud Run dosya sistemi efemeral; bu projede runtime yazma yoktur.
- Modeli yeniden eğitmek için `scripts/train_model.py` lokal çalıştırılır, çıktılar (`api/taxi_demand_model.pkl`, `api/zone_stats.json`) commit edilip yeniden deploy edilir.

## Cloud Build Süresi Uyarısı
`taxi-api` imajı XGBoost, scikit-learn, pandas ve numpy içerir; ilk build **5–8 dakika** sürebilir ve nihai imaj ~1 GB civarındadır. Sonraki build'ler katman önbelleği sayesinde daha hızlıdır.
