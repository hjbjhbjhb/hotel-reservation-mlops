"""
Week 3 — Model Training & Evaluation
Hotel Reservation Cancellation Prediction

Trains three models, evaluates each, and saves the best one.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

# Local imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_data, preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:12s}: {v}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Canceled", "Canceled"]))

    return metrics


def feature_importance(model, feature_names: list, top_n: int = 10) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False).head(top_n)
    print("\n  Top Feature Importances:")
    for feat, val in imp.items():
        bar = "█" * int(val * 50)
        print(f"  {feat:45s} {val:.4f}  {bar}")


def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    all_metrics = []
    trained = {}

    for name, model in models.items():
        print(f"\n[INFO] Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        metrics = evaluate(name, model, X_test, y_test)
        feature_importance(model, feature_names)
        all_metrics.append(metrics)

    # Summary table
    results_df = pd.DataFrame(all_metrics).set_index("model")
    print("\n\n===== MODEL COMPARISON =====")
    print(results_df.to_string())

    # Save results
    results_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"))
    print(f"\n[INFO] Results saved to outputs/model_comparison.csv")

    # Save best model (by ROC-AUC)
    best_name = results_df["roc_auc"].idxmax()
    best_model = trained[best_name]
    best_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(best_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"[INFO] Best model: {best_name} (ROC-AUC = {results_df.loc[best_name,'roc_auc']})")
    print(f"[INFO] Saved to models/best_model.pkl")

    return results_df


if __name__ == "__main__":
    train_and_evaluate()
