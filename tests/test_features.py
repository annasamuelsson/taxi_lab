from taxi_fare.features import build_features
import pandas as pd

def test_build_features():
    df = pd.DataFrame([{
        "pickup_lat": 59.33, "pickup_lon": 18.06,
        "dropoff_lat": 59.36, "dropoff_lon": 18.01,
        "pickup_datetime": "2025-01-01T10:00:00Z",
        "fare_amount": 140.0
    }])
    mapping = {
        "pickup_lat":"pickup_lat","pickup_lon":"pickup_lon",
        "dropoff_lat":"dropoff_lat","dropoff_lon":"dropoff_lon"
    }
    X = build_features(df, mapping, "pickup_datetime")
    assert list(X.columns) == ["dist","hour"]
    assert len(X) == 1
