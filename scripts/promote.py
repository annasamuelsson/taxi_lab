from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
import os
import sys

# --- bootstrap: kör alltid från repo-roten och fixa PYTHONPATH ---
import os, sys, subprocess
from pathlib import Path

try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Fallback om __file__ inte finns (t.ex. i notebook / REPL)
    REPO_ROOT = Path.cwd()
os.chdir(REPO_ROOT)  # gör repo-roten till CWD

SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

def ensure_installed():
    try:
        import pandas  # snabb koll
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements.txt")])
#    try:
#        import taxi_fare  # noqa
#    except Exception:
#        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(REPO_ROOT)])

ensure_installed()
# --- end bootstrap ---


MODEL_NAME = os.environ.get("MODEL_NAME", "taxi_fare_model")

client = MlflowClient()

def get_latest_version(name: str):
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise ValueError(f"No versions found for model: {name}")
    latest = max(versions, key=lambda v: int(v.version))
    return latest

def get_metric(run_id: str, metric_name: str):
    r = client.get_run(run_id)
    return r.data.metrics.get(metric_name)

def main():
    latest = get_latest_version(MODEL_NAME)
    latest_mae = get_metric(latest.run_id, "mae_holdout")
    print(f"Latest: v{latest.version} (run {latest.run_id}) mae={latest_mae}")

    prod_mae = None
    try:
        prod_list = client.get_latest_versions(name=MODEL_NAME, stages=["Production"])
        if prod_list:
            prod_ver = prod_list[0]
            prod_mae = get_metric(prod_ver.run_id, "mae_holdout")
            print(f"Production: v{prod_ver.version} mae={prod_mae}")
        else:
            print("No Production version.")
    except RestException as e:
        print("Error fetching Production:", e)

    should_promote = prod_mae is None or (latest_mae is not None and latest_mae < prod_mae)
    if should_promote:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Promoted {MODEL_NAME} v{latest.version} → Production")
    else:
        print("Not promoting (no improvement).")

if __name__ == "__main__":
    sys.exit(main() or 0)
