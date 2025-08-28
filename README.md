# Taxi Fare Service (MLOps-ready)

End-to-end demo for ML/MLOps & Databricks:
- `src/taxi_fare` – pure Python package (features, data, model, predict)
- `scripts/train.py` – trains and logs with MLflow
- `app/main.py` – FastAPI inference API (optional if you use Databricks Model Serving)
- `configs/` – YAML config for training/app
- `tests/` – minimal pytest suite
- `docker/` – Dockerfile to run FastAPI
- `.github/workflows/ci.yml` – lint/test + docker build

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/train.py --config configs/training.yaml
uvicorn app.main:app --reload --port 8080
```

## Databricks usage
- Add this repo via **Databricks Repos**.
- Create a notebook that imports and calls `scripts/train.py` or functions in `src/taxi_fare`.
- Use MLflow UI/Model Registry in Databricks; enable Model Serving for the registered model.
