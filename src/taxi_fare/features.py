import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame,
                   mapping: dict,
                   datetime_col: str) -> pd.DataFrame:
    # Rename to canonical names using mapping
    df = df.rename(columns=mapping).copy()
    # Basic engineered features
    df["dist"] = np.sqrt((df["dropoff_lat"]-df["pickup_lat"])**2 +
                         (df["dropoff_lon"]-df["pickup_lon"])**2)
    df["hour"] = pd.to_datetime(df[datetime_col]).dt.hour
    return df[["dist","hour"]]
