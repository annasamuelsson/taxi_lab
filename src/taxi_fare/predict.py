from typing import Dict, Any
import pandas as pd
from .model import load_model

def predict_single(model, payload: Dict[str, Any]) -> float:
    # Expect features already engineered externally; keep a minimal fallback:
    if {"dist","hour"} <= set(payload.keys()):
        X = pd.DataFrame([{"dist": payload["dist"], "hour": payload["hour"]}])
    else:
        # If raw coords/datetime given:
        from .features import build_features
        import pandas as pd
        df = pd.DataFrame([payload])
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        X = build_features(df, {
            "pickup_lat": "pickup_lat",
            "pickup_lon": "pickup_lon",
            "dropoff_lat": "dropoff_lat",
            "dropoff_lon": "dropoff_lon",
        }, "pickup_datetime")
    y = model.predict(X)[0]
    return float(y)

def load_model_from_path(path: str):
    return load_model(path)
