# run_job.py
import sys, subprocess
from pathlib import Path
import runpy

# Uppgradera innan något annan hinner importera sklearn/numpy
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "threadpoolctl>=3.5.1"])

# (valfritt, om du vill säkra versionskombo)
# subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "scikit-learn>=1.3", "numpy>=1.25"])

# Kör train.py
runpy.run_module("train", run_name="__main__")
