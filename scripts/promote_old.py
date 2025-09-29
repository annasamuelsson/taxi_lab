import os
import sys
from pathlib import Path

# --- bootstrap: hitta repo-roten, fixa sys.path och beroenden ---
import subprocess

def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists() or (p / "configs").exists():
            return p
    return start

try:
    here = Path(__file__).resolve()
except NameError:
    here = Path.cwd()

repo_root = find_repo_root(here.parent)  # scripts/ -> repo/
os.chdir(repo_root)

src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def ensure_installed():
    # vi behöver främst mlflow här; installera requirements om det saknas
    try:
        import mlflow  # noqa
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(repo_root / "requirements.txt")])
    # (Projektet i sig behöver inte installeras för promote, men det skadar inte:)
    try:
        import taxi_fare  # noqa
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(repo_root)])
        except Exception:
            # ok att fortsätta – promote använder inte paketet direkt
            pass

ensure_installed()
# --- end bootstrap ---

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

#MODEL_NAME = os.environ.get("MODEL_NAME", "taxi_fare_model")

def resolve_model_name(base_name: str) -> str:
    uri = mlflow.get_registry_uri()
    if uri and uri.startswith("databricks-uc") and base_name.count(".") != 2:
        cat = os.getenv("UC_CATALOG", "main")
        sch = os.getenv("UC_SCHEMA", "default")
        return f"{cat}.{sch}.{base_name}"
    return base_name

BASE_MODEL_NAME = os.environ.get("MODEL_NAME", "taxi_fare_model")
MODEL_NAME = resolve_model_name(BASE_MODEL_NAME)

client = MlflowClient()


def get_latest_version(name: str):
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise ValueError(f"No versions found for model: {name}")
    return max(versions, key=lambda v: int(v.version))


def get_metric(run_id: str, metric_name: str):
    r = client.get_run(run_id)
    return r.data.metrics.get(metric_name)


def main() -> int:
    latest = get_latest_version(MODEL_NAME)
    latest_mae = get_metric(latest.run_id, "mae_holdout")
    print(f"Latest candidate: v{latest.version} (run {latest.run_id}) mae_holdout={latest_mae}")

    prod_mae = None
    prod_ver = None
    try:
        prod_list = client.get_latest_versions(name=MODEL_NAME, stages=["Production"])
        if prod_list:
            prod_ver = prod_list[0]
            prod_mae = get_metric(prod_ver.run_id, "mae_holdout")
            print(f"Current Production: v{prod_ver.version} mae_holdout={prod_mae}")
        else:
            print("No Production version yet.")
    except RestException as e:
        print("Error fetching Production version:", e)

    should_promote = (prod_mae is None) or (latest_mae is not None and prod_mae is not None and latest_mae < prod_mae)

    if should_promote:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Promoted {MODEL_NAME} v{latest.version} → Production (archived previous).")
        return 0
    else:
        print("Not promoting: latest candidate is not better than Production.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
