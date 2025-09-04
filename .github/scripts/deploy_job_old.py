import os, sys, json, urllib.request

HOST = os.environ.get("DATABRICKS_HOST")
TOKEN = os.environ.get("DATABRICKS_TOKEN")
if not HOST or not TOKEN:
    print("Missing DATABRICKS_HOST or DATABRICKS_TOKEN", file=sys.stderr)
    sys.exit(1)

JOB_NAME = os.environ.get("JOB_NAME", "taxi-fare-train")
REPO_URL = os.environ.get("REPO_URL")
GH_REPO = os.environ.get("GITHUB_REPOSITORY")
if not REPO_URL and GH_REPO:
    REPO_URL = f"https://github.com/{GH_REPO}.git"
BRANCH = os.environ.get("BRANCH") or os.environ.get("GITHUB_REF_NAME") or "main"
EXISTING_CLUSTER_ID = os.environ.get("EXISTING_CLUSTER_ID")  # optional
MODEL_NAME = os.environ.get("MODEL_NAME", "taxi_fare_model")

headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

def api(path, payload=None, method="GET"):
    url = HOST.rstrip("/") + path
    req = urllib.request.Request(url, headers=headers, method=method)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req.data = data
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

# Build tasks
train_task = {
    "task_key": "train",
    "description": "Train taxi fare model and log to MLflow",
    "notebook_task": {"notebook_path": "notebooks/train_job.py", "source": "GIT"},
    "timeout_seconds": 0,
    "email_notifications": {},
}

promote_task = {
    "task_key": "promote",
    "description": "Promote latest model to Production if MAE improved",
    "notebook_task": {
        "notebook_path": "notebooks/promote_job.py",
        "base_parameters": {"model_name": MODEL_NAME},
        "source": "GIT",
    },
    "timeout_seconds": 0,
    "depends_on": [{"task_key": "train"}],
}

settings = {
    "name": JOB_NAME,
    "max_concurrent_runs": 1,
    "git_source": {
        "git_url": REPO_URL,
        "git_branch": BRANCH
    },
    "tasks": [train_task, promote_task],
}

if EXISTING_CLUSTER_ID:
    train_task["existing_cluster_id"] = EXISTING_CLUSTER_ID
    promote_task["existing_cluster_id"] = EXISTING_CLUSTER_ID
else:
    settings["job_clusters"] = [{
        "job_cluster_key": "single_node",
        "new_cluster": {
            "spark_version": os.environ.get("SPARK_VERSION", "13.3.x-scala2.12"),
            "spark_conf": {
                "spark.master": "local[*]",
                "spark.databricks.cluster.profile": "singleNode"
            },
            "num_workers": 0,
            "node_type_id": os.environ.get("NODE_TYPE_ID", "Standard_D4as_v5")
        }
    }]
    train_task["job_cluster_key"] = "single_node"
    promote_task["job_cluster_key"] = "single_node"

# List existing jobs
jobs = api("/api/2.1/jobs/list").get("jobs", [])
job = next((j for j in jobs if j.get("settings", {}).get("name") == JOB_NAME), None)

if job:
    job_id = job["job_id"]
    print(f"Updating existing job {JOB_NAME} (id={job_id})")
    payload = {"job_id": job_id, "new_settings": settings}
    api("/api/2.1/jobs/reset", payload=payload, method="POST")
    print("Job reset successful.")
else:
    print(f"Creating new job {JOB_NAME}")
    payload = settings
    created = api("/api/2.1/jobs/create", payload=payload, method="POST")
    print("Job created with id:", created.get("job_id"))
