import argparse
import yaml
from pathlib import Path
import mlflow
import mlflow.sklearn
from taxi_fare.data import load_training_data
from taxi_fare.features import build_features
from taxi_fare.model import train_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
#from evidently.report import Report
#from evidently.metric_preset import DataDriftPreset
from evidently import Report
from evidently.metric_preset import DataDriftPreset


def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
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
        report = Report(metrics=[DataDriftPreset()])
        # Build small dataframes to compare distributional drift on the features
        import pandas as pd
        df_train = pd.DataFrame(X_train, columns=X.columns)
        df_test = pd.DataFrame(X_test, columns=X.columns)
        report.run(reference_data=df_train, current_data=df_test)
        report_path = artifacts_dir.parent / "evidently_data_drift_report.html"
        report.save_html(str(report_path))
        mlflow.log_artifact(str(report_path))

        print(f"Saved model → {model_path} | MAE={mae:.4f} | Evidently report logged: {report_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    args = ap.parse_args()
    main(args.config)
