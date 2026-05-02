"""
Train, evaluate, and save phishing detection models.
"""
import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from features import extract_features

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "phishing_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_extract(path: str):
    print("📂  Loading dataset …")
    df = pd.read_csv(path)
    print(f"    Rows: {len(df)}  |  Phishing: {df['label'].sum()}  |  Legit: {(df['label']==0).sum()}")

    print("🔍  Extracting features …")
    feature_rows = df["url"].apply(lambda u: pd.Series(extract_features(u)))
    X = feature_rows.values
    y = df["label"].values
    feature_names = feature_rows.columns.tolist()
    print(f"    Feature count: {len(feature_names)}")
    return X, y, feature_names


def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    metrics = dict(
        accuracy  = accuracy_score(y_test, y_pred),
        precision = precision_score(y_test, y_pred, zero_division=0),
        recall    = recall_score(y_test, y_pred, zero_division=0),
        f1        = f1_score(y_test, y_pred, zero_division=0),
        confusion  = confusion_matrix(y_test, y_pred).tolist(),
    )

    print(f"\n{'═'*55}")
    print(f"  {name}")
    print(f"{'═'*55}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = np.array(metrics["confusion"])
    print(f"               Predicted")
    print(f"             Legit  Phishing")
    print(f"  Legit     {cm[0][0]:5d}    {cm[0][1]:5d}")
    print(f"  Phishing  {cm[1][0]:5d}    {cm[1][1]:5d}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Legitimate", "Phishing"],
                                zero_division=0))
    return metrics


def train_all():
    X, y, feature_names = load_and_extract(DATA_PATH)

    # ── Train / test split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Scaling (Logistic Regression benefits from scaling) ────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Model definitions ──────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42
        ),
        "Random Forest (Primary)": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            min_samples_split=2, random_state=42, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20, min_samples_split=5, random_state=42
        ),
    }

    results = {}
    trained = {}

    print("\n" + "="*55)
    print("  🤖  TRAINING PHISHING DETECTION MODELS")
    print("="*55)

    for name, clf in models.items():
        print(f"\n⚙️  Training {name} …")
        if name == "Logistic Regression":
            clf.fit(X_train_scaled, y_train)
            metrics = evaluate(name, clf, X_test_scaled, y_test)
        else:
            clf.fit(X_train, y_train)
            metrics = evaluate(name, clf, X_test, y_test)
        results[name] = metrics
        trained[name] = clf

    # ── Comparison table ───────────────────────────────────────────
    print("\n" + "="*55)
    print("  📊  MODEL COMPARISON SUMMARY")
    print("="*55)
    print(f"  {'Model':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-'*54}")
    for name, m in results.items():
        print(f"  {name:<30} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
              f"{m['recall']:>6.4f} {m['f1']:>6.4f}")

    # ── Save primary model (Random Forest) ────────────────────────
    primary_model = trained["Random Forest (Primary)"]
    joblib.dump(primary_model, os.path.join(MODEL_DIR, "phishing_model.pkl"))
    joblib.dump(scaler,        os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(trained["Logistic Regression"],
                os.path.join(MODEL_DIR, "logistic_model.pkl"))

    # ── Save metadata ──────────────────────────────────────────────
    meta = {
        "feature_names": feature_names,
        "model_name": "Random Forest",
        "num_features": len(feature_names),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "confusion"}
                    for k, v in results.items()},
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅  Models saved to '{MODEL_DIR}'")
    return trained, scaler, feature_names, results


if __name__ == "__main__":
    train_all()
