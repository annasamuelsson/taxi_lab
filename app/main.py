from fastapi import FastAPI, Request
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from typing import Optional
import yaml
from pathlib import Path

from src.taxi_fare.predict import predict_single, load_model_from_path

app = FastAPI(title="Taxi Fare Service")
# Prometheus metrics
PREDICTIONS_TOTAL = Counter('predictions_total', 'Total number of predictions served')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Latency of prediction endpoint')

# Config
cfg_path = Path("configs/app.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
MODEL_PATH = cfg.get("model_path", "artifacts/models/model.joblib")

# Load model at startup
model = None
@app.on_event("startup")
def _load_model():
    global model
    if Path(MODEL_PATH).exists():
        model = load_model_from_path(MODEL_PATH)
    else:
        model = None

class RawRequest(BaseModel):
    pickup_lat: float
    pickup_lon: float
    dropoff_lat: float
    dropoff_lon: float
    pickup_datetime: str

class FeatureRequest(BaseModel):
    dist: float
    hour: int

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(req: RawRequest, request: Request):
    if model is None:
        return {"error": "Model not loaded. Train first."}
    start = time.time()
    y = predict_single(model, req.dict())
    duration = time.time() - start
    PREDICTIONS_TOTAL.inc()
    PREDICTION_LATENCY.observe(duration)
    return {"fare": y}

@app.post("/predict_features")
def predict_features(req: FeatureRequest, request: Request):
    if model is None:
        return {"error": "Model not loaded. Train first."}
    start = time.time()
    y = predict_single(model, req.dict())
    duration = time.time() - start
    PREDICTIONS_TOTAL.inc()
    PREDICTION_LATENCY.observe(duration)
    return {"fare": y}


@app.get('/metrics')
def metrics():
    return FastAPI.responses.Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
