import joblib
import pandas as pd
from .random_forest_model import FEATURE_NAMES
from .train_model import MODEL_PATH, ensure_model


def predict_suitability(features: dict) -> dict:
    ensure_model()
    model = joblib.load(MODEL_PATH)
    values = pd.DataFrame([[float(features.get(name, 0)) for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    probability = float(model.predict_proba(values)[0][1] * 100)
    label = "HIGH POTENTIAL" if probability >= 70 else "MODERATE POTENTIAL" if probability >= 45 else "LOW POTENTIAL"
    return {"probability": round(probability, 2), "label": label}
