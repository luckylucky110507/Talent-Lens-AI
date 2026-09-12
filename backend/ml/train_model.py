import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from .random_forest_model import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_PATH = DATA_DIR / "candidate_training_data.csv"
MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
METRICS_PATH = MODEL_DIR / "random_forest_metrics.json"


def create_dataset(n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.uniform(18, 98, size=(n, 7))
    skill, keyword, similarity, experience, education, project, certification = data.T
    ats = skill * .35 + keyword * .25 + similarity * .20 + (experience + education) / 2 * .10 + (project + certification) / 2 * .10
    signal = .34 * skill + .24 * similarity + .18 * experience + .14 * project + .10 * certification + rng.normal(0, 8, n)
    target = (signal >= 57).astype(int)
    frame = pd.DataFrame(data, columns=FEATURE_NAMES[:7])
    frame["ats_score"] = ats
    frame["candidate_suitable"] = target
    DATA_DIR.mkdir(exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)
    return frame


def train_model() -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    frame = create_dataset()
    X = frame[FEATURE_NAMES]
    y = frame["candidate_suitable"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_leaf=3, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        "test_size": len(y_test),
        "training_rows": len(frame),
    }
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    joblib.dump({"feature_names": FEATURE_NAMES, "importance": model.feature_importances_.tolist()}, MODEL_DIR / "feature_importance.joblib")
    return metrics


def ensure_model() -> dict:
    if not MODEL_PATH.exists() or not METRICS_PATH.exists() or not DATA_PATH.exists():
        return train_model()
    return json.loads(METRICS_PATH.read_text())
