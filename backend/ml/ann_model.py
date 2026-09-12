import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .random_forest_model import FEATURE_NAMES
from .train_model import create_dataset

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
ANN_MODEL_PATH = MODEL_DIR / "ann_model.joblib"
ANN_META_PATH = MODEL_DIR / "ann_metrics.json"


def train_ann() -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    frame = create_dataset()
    X = frame[FEATURE_NAMES].to_numpy()
    y = frame["candidate_suitable"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # This is the academic fallback for environments where TensorFlow has no Python 3.13 wheel.
    model = MLPClassifier(hidden_layer_sizes=(32, 16, 8), activation="relu", solver="adam", max_iter=450, random_state=42, early_stopping=True, validation_fraction=.15)
    model.fit(X_train, y_train)
    accuracy = float(model.score(X_test, y_test))
    payload = {"model": model, "scaler": scaler, "architecture": ["Input(8)", "Dense(32, relu)", "Dense(16, relu)", "Dense(8, relu)", "Dense(1, sigmoid)"], "backend": "scikit-learn MLPClassifier (ANN-compatible fallback)"}
    joblib.dump(payload, ANN_MODEL_PATH)
    metrics = {"accuracy": round(accuracy, 4), "validation_split": .15, "backend": payload["backend"], "architecture": payload["architecture"]}
    ANN_META_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def ensure_ann() -> dict:
    if not ANN_MODEL_PATH.exists() or not ANN_META_PATH.exists():
        return train_ann()
    return json.loads(ANN_META_PATH.read_text())
