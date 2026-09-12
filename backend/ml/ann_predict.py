import joblib
import numpy as np
from .ann_model import ANN_MODEL_PATH, ensure_ann
from .random_forest_model import FEATURE_NAMES


def predict_ann(features: dict) -> dict:
    ensure_ann()
    payload = joblib.load(ANN_MODEL_PATH)
    values = np.array([[float(features.get(name, 0)) for name in FEATURE_NAMES]])
    probability = float(payload["model"].predict_proba(payload["scaler"].transform(values))[0][1] * 100)
    label = "HIGH POTENTIAL" if probability >= 70 else "MODERATE POTENTIAL" if probability >= 45 else "LOW POTENTIAL"
    return {"probability": round(probability, 2), "label": label, "backend": payload["backend"]}
