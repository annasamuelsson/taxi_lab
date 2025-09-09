# Databricks notebook source
dbutils.widgets.text("model_name", "taxi_fare_model", "Model name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

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

latest = get_latest_version(model_name)
latest_mae = get_metric(latest.run_id, "mae_holdout")
print(f"Latest version: v{latest.version}, run_id={latest.run_id}, mae_holdout={latest_mae}")

prod_mae = None
try:
    prod_list = client.get_latest_versions(name=model_name, stages=["Production"])
    if prod_list:
        prod_ver = prod_list[0]
        prod_mae = get_metric(prod_ver.run_id, "mae_holdout")
        print(f"Current Production: v{prod_ver.version}, run_id={prod_ver.run_id}, mae_holdout={prod_mae}")
    else:
        print("No Production version exists yet.")
except RestException as e:
    print("Error fetching Production version:", e)

should_promote = False
if prod_mae is None:
    should_promote = True
elif latest_mae is not None and latest_mae < prod_mae:
    should_promote = True

if should_promote:
    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Promoted {model_name} v{latest.version} to Production (archived previous).")
else:
    print("Not promoting: latest MAE is not better than Production.")
