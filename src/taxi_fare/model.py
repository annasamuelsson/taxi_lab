from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor

def train_model(X, y, **model_params):
    model = RandomForestRegressor(**model_params)
    model.fit(X, y)
    return model

def save_model(model, path: str):
    dump(model, path)

def load_model(path: str):
    return load(path)
