# Databricks notebook to run training from repo package
# Keep thin: just call the script's main or directly import functions
# If running as a notebook task, the working directory is the repo checkout root.
# You can also %pip install -r requirements.txt if using a fresh job cluster.
# Use dbutils.widgets to pass config path if desired.

# COMMAND ----------
# %pip install -r requirements.txt

# COMMAND ----------
import os
os.system("python scripts/train.py --config configs/training.yaml")
print("Training completed.")
