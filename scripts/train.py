import argparse
import yaml
from pathlib import Path
import mlflow
import mlflow.sklearn
from src.taxi_fare.data import load_training_data
from src.taxi_fare.features import build_features
from src.taxi_fare.model import train_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
#from evidently.report import Report
#from evidently.metric_preset import DataDriftPreset

# överst, ersätt imports med:
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_OK = True
except Exception:
    EVIDENTLY_OK = False

# --- bootstrap: kör alltid från repo-roten och fixa PYTHONPATH ---
import os, sys, subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../repo
os.chdir(REPO_ROOT)  # gör repo-roten till CWD

SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

def ensure_installed():
    try:
        import pandas  # snabb koll
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements.txt")])
    try:
        import taxi_fare  # noqa
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(REPO_ROOT)])
ensure_installed()
# --- end bootstrap ---

def main(config_path: str):
    cfg_path = (REPO_ROOT / config_path) if not Path(config_path).is_file() else Path(config_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found. Tried: {config_path} and {REPO_ROOT / config_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    data_path = cfg["data_path"]
    target_col = cfg["target_col"]
    datetime_col = cfg["datetime_col"]
    mapping = cfg["feature_mapping"]
    model_params = cfg["model_params"]
    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts/models"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data(data_path)
    X = build_features(df, mapping, datetime_col)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri(cfg.get("mlflow_uri", "file:./mlruns"))
    mlflow.set_experiment(cfg.get("experiment_name", "taxi_fare_experiment"))
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    with mlflow.start_run(run_name="rf_regressor"):
        model = train_model(X_train, y_train, **model_params)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mlflow.log_metric("mae_holdout", mae)

        model_path = artifacts_dir / "model.joblib"
        save_model(model, str(model_path))
        mlflow.log_artifact(str(model_path))

        # ---- Monitoring hook: generate Evidently data drift report (train vs test) ----
        #report = Report(metrics=[DataDriftPreset()])
        # Build small dataframes to compare distributional drift on the features
        #import pandas as pd
        #df_train = pd.DataFrame(X_train, columns=X.columns)
        #df_test = pd.DataFrame(X_test, columns=X.columns)
        #report.run(reference_data=df_train, current_data=df_test)
        #report_path = artifacts_dir.parent / "evidently_data_drift_report.html"
        #report.save_html(str(report_path))
        #mlflow.log_artifact(str(report_path))

        if EVIDENTLY_OK:
            from pandas import DataFrame
            df_train = DataFrame(X_train, columns=X.columns)
            df_test = DataFrame(X_test, columns=X.columns)
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=df_train, current_data=df_test)
            report_path = artifacts_dir.parent / "evidently_data_drift_report.html"
            report.save_html(str(report_path))
            mlflow.log_artifact(str(report_path))
            print(f"Saved model → {model_path} | MAE={mae:.4f} | Evidently report logged: {report_path}")
        else:
            print("Evidently saknas/inkompatibelt – hoppar över drift-rapporten.")
            print(f"Saved model → {model_path} | MAE={mae:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    args = ap.parse_args()
    main(args.config)
