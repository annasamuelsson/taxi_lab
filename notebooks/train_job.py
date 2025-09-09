# Databricks notebook source
# MAGIC %pip install -r requirements.txt

# COMMAND ----------
import os
exit_code = os.system("python scripts/train.py --config configs/training.yaml")
if exit_code != 0:
    raise SystemExit(f"Training script failed with exit code {exit_code}")
print("Training completed.")