import argparse, yaml
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error
from src.taxi_fare.predict import load_model_from_path, predict_single

def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    model_path = cfg.get("artifacts_dir", "artifacts/models") + "/model.joblib"

    # Simple holdout based on the same sample CSV (just demo)
    df = pd.read_csv("data/sample.csv")
    model = load_model_from_path(model_path)

    preds = []
    for _, r in df.iterrows():
        preds.append(predict_single(model, r.to_dict()))
    mae = mean_absolute_error(df["fare_amount"], preds)
    print(f"Holdout MAE: {mae:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    args = ap.parse_args()
    main(args.config)
