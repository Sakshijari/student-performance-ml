"""
Train and evaluate multiple classifiers for student performance prediction.
Models: Logistic Regression, Random Forest, SVM, Gradient Boosting.
Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
"""

import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42


def get_models():
    """Return dict of model name -> sklearn estimator (with class_weight for imbalance)."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        ),
    }


def evaluate_model(model, X_test, y_test, name="Model"):
    """Compute metrics and confusion matrix for a fitted model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    else:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return {"metrics": metrics, "confusion_matrix": cm.tolist(), "classification_report": report}


def run_training_and_evaluation(data_result, output_dir: str = None):
    """
    Train all models, evaluate on test set, optionally run cross-validation and save artefacts.
    data_result: dict from preprocess_and_split().
    """
    X_train = data_result["X_train"]
    X_val = data_result["X_val"]
    X_test = data_result["X_test"]
    y_train = data_result["y_train"]
    y_val = data_result["y_val"]
    y_test = data_result["y_test"]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

    results = {}
    models = get_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        ev = evaluate_model(model, X_test, y_test, name)
        results[name] = ev
        if output_dir:
            joblib.dump(model, os.path.join(models_dir, f"{name.replace(' ', '_')}.joblib"))

    # Cross-validation (F1) for robustness
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        cv_scores[name] = {"mean_f1": float(scores.mean()), "std_f1": float(scores.std())}
    results["cross_validation"] = cv_scores

    if output_dir:
        to_save = {k: v for k, v in results.items() if k != "cross_validation"}
        to_save["cross_validation"] = cv_scores
        with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
            json.dump(to_save, f, indent=2)

    return results


def print_results_summary(results):
    """Print a compact comparison table and CV summary."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Test Set)")
    print("=" * 60)
    for name in ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting"]:
        if name not in results:
            continue
        m = results[name]["metrics"]
        print(f"\n{name}:")
        print(f"  Accuracy: {m['accuracy']:.4f}  Precision: {m['precision']:.4f}")
        print(f"  Recall:   {m['recall']:.4f}  F1: {m['f1']:.4f}  ROC-AUC: {m.get('roc_auc') or 'N/A'}")
        print(f"  Confusion Matrix: {results[name]['confusion_matrix']}")

    if "cross_validation" in results:
        print("\n" + "-" * 60)
        print("5-Fold Stratified CV (F1)")
        for name, cv in results["cross_validation"].items():
            print(f"  {name}: {cv['mean_f1']:.4f} (+/- {cv['std_f1']:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import load_data, preprocess_and_split

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "StudentsPerformance.csv")
    df = load_data(data_path)
    data_result = preprocess_and_split(df)
    output_dir = os.path.join(base_dir, "output")
    results = run_training_and_evaluation(data_result, output_dir=output_dir)
    print_results_summary(results)
