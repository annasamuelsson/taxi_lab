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

# === Konfiguration ===
PROD_ALIAS = os.getenv("PROD_ALIAS", "prod")  # UC: använd alias som motsvarighet till "Production"

def resolve_model_name(base_name: str) -> str:
    """
    Om vi kör mot Unity Catalog (registry_uri börjar med 'databricks-uc')
    och namnet inte redan är 'catalog.schema.name', så prefixa.
    """
    uri = mlflow.get_registry_uri()
    if uri and uri.startswith("databricks-uc") and base_name.count(".") != 2:
        cat = os.getenv("UC_CATALOG", "main")
        sch = os.getenv("UC_SCHEMA", "default")
        return f"{cat}.{sch}.{base_name}"
    return base_name

BASE_MODEL_NAME = os.environ.get("MODEL_NAME", "taxi_fare_model")
MODEL_NAME = resolve_model_name(BASE_MODEL_NAME)

client = MlflowClient()


def get_latest_ready_version(name: str):
    """
    UC stödjer inte stage-baserade 'latest'. Vi hämtar senaste READY-versionen manuellt.
    """
    versions = list(client.search_model_versions(f"name='{name}'"))
    if not versions:
        raise ValueError(f"No versions found for model: {name}")
    ready = [v for v in versions if getattr(v, "status", "READY") == "READY"]
    candidates = ready if ready else versions
    return max(candidates, key=lambda v: int(v.version))


def get_version_by_alias(name: str, alias: str):
    """
    Hämta model version som alias pekar på (UC).
    """
    try:
        return client.get_model_version_by_alias(name=name, alias=alias)
    except Exception:
        return None


def get_metric(run_id: str, metric_name: str):
    r = client.get_run(run_id)
    return r.data.metrics.get(metric_name)


def main() -> int:
    # Kandidat = senaste READY-version
    latest = get_latest_ready_version(MODEL_NAME)
    latest_mae = get_metric(latest.run_id, "mae_holdout")
    print(f"Latest candidate: {MODEL_NAME} v{latest.version} (run {latest.run_id}) mae_holdout={latest_mae}")

    # Nuvarande "prod" via alias
    prod_mv = get_version_by_alias(MODEL_NAME, PROD_ALIAS)
    prod_mae = None
    if prod_mv:
        prod_mae = get_metric(prod_mv.run_id, "mae_holdout")
        print(f"Current @{PROD_ALIAS}: v{prod_mv.version} (run {prod_mv.run_id}) mae_holdout={prod_mae}")
    else:
        print(f"No alias '@{PROD_ALIAS}' set yet.")

    # Om alias redan pekar på latest-versionen – inget att göra
    if prod_mv and int(prod_mv.version) == int(latest.version):
        print(f"Alias '@{PROD_ALIAS}' already points to v{latest.version}. Nothing to do.")
        return 0

    # Promotionsvillkor
    should_promote = (prod_mae is None) or (
        latest_mae is not None and prod_mae is not None and latest_mae < prod_mae
    )

    if should_promote:
        # UC: sätt alias i stället för stages
        try:
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias=PROD_ALIAS,
                version=str(latest.version),
            )
            print(f"Promoted {MODEL_NAME} v{latest.version} → @{PROD_ALIAS} (alias updated).")
        except RestException as e:
            print("Error setting alias:", e)
            return 1
        return 0
    else:
        print("Not promoting: latest candidate is not better than current alias.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
