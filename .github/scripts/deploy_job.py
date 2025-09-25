# --- Tasks as scripts instead of notebooks ---
train_task = {
    "task_key": "train",
    "description": "Train taxi fare model and log to MLflow",
    "spark_python_task": {
        "python_file": "scripts/train.py",
        "parameters": ["--config", "configs/training.yaml"]
    },
    "timeout_seconds": 0,
}

promote_task = {
    "task_key": "promote",
    "description": "Promote latest model to Production if MAE improved",
    "spark_python_task": {
        "python_file": "scripts/promote.py"
    },
    "timeout_seconds": 0,
    "depends_on": [{"task_key": "train"}],
}
