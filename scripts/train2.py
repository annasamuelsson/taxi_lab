# train.py — MLflow hanterar alla artefakter (ingen lokal artifacts-dir)

# (valfritt om du kör via run_job.py som redan installerar rätt version)
try:
    import threadpoolctl
    from packaging.version import Version
    if Version(threadpoolctl.__version__) < Version("3.5.1"):
        raise ImportError
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "threadpoolctl>=3.5.1"])

import argparse
import yaml
from pathlib import Path
import tempfile
import os, sys, subprocess

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- bootstrap: hitta repo-roten och fixa sys.path ---
def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists() or (p / "configs").exists():
            return p
    return start

try:
    here = Path(__file__).resolve()
except NameError:
    here = Path.cwd()

repo_root = find_repo_root(here.parent)   # scripts/ -> repo/
os.chdir(repo_root)

src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Nu kan vi importera vårt paket
from taxi_fare.data import load_training_data
from taxi_fare.features import build_features
from taxi_fare.model import train_model, save_model  # save_model används ej, men låter den vara kvar

# Evidently (valfritt)
EVIDENTLY_OK = False
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_OK = True
except Exception:
    EVIDENTLY_OK = False


def main(config_path: str):
    # Resolva config-path (absolut eller relativt repo-roten)
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        cfg_path = (repo_root / config_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found. Tried: {config_path} and {repo_root / config_path}")

    cfg = yaml.safe_load(cfg_path.read_text())
    data_path = cfg["data_path"]
    target_col = cfg.get("target_col", "fare_amount")
    datetime_col = cfg.get("datetime_col", "pickup_datetime")
    mapping = cfg["feature_mapping"]
    model_params = cfg["model_params"]

    # Data & features
    df = load_training_data(data_path)
    X = build_features(df, mapping, datetime_col)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.get("test_size", 0.2), random_state=cfg.get("random_state", 42)
    )

    # MLflow setup — logga direkt till Databricks MLflow
    mlflow.set_tracking_uri(cfg.get("mlflow_uri", "databricks"))
    mlflow.set_experiment(cfg.get("experiment_name", "/Shared/taxi_fare_experiment"))
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    with mlflow.start_run(run_name="rf_regressor"):
        model = train_model(X_train, y_train, **model_params)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mlflow.log_metric("mae_holdout", mae)

        # ⬇️ Nyckeln i Alternativ A: logga modellen direkt till MLflow (ingen lokal mapp)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Evidently-rapport (om Evidently finns): skriv till /tmp och logga till MLflow
        report_path = None
        if EVIDENTLY_OK:
            try:
                from pandas import DataFrame
                df_train = DataFrame(X_train, columns=X.columns)
                df_test = DataFrame(X_test, columns=X.columns)
                report = Report(metrics=[DataDriftPreset()])
                report.run(reference_data=df_train, current_data=df_test)
                tmp_dir = Path(tempfile.mkdtemp(prefix="taxi_fare_"))
                report_path = tmp_dir / "evidently_data_drift_report.html"
                report.save_html(str(report_path))
                mlflow.log_artifact(str(report_path))
            except Exception as e:
                print(f"Evidently misslyckades, hoppar över rapport. Orsak: {e}")

        # Sammanfattning
        if report_path is not None:
            print(f"MLflow: model logged under 'model' | MAE={mae:.4f} | Evidently report logged")
        else:
            print(f"MLflow: model logged under 'model' | MAE={mae:.4f} | Evidently report: skipped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    args = ap.parse_args()
    main(args.config)
