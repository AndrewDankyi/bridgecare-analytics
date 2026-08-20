"""
train_model.py

Trains and compares four classifiers (Random Forest, XGBoost, Logistic
Regression, Decision Tree) to predict 30-day hospital readmission,
balances the training set with SMOTE, evaluates on a held-out test set,
and generates SHAP feature importances for the winning model.

Usage:
    python generate_sample_data.py   # creates data/sample_readmission_data.csv
    python train_model.py            # trains, evaluates, saves outputs/
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data") / "sample_readmission_data.csv"
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

NUMERIC_FEATURES = [
    "time_in_hospital", "num_prior_inpatient", "num_diagnoses",
    "num_medications", "num_lab_procedures",
]
CATEGORICAL_FEATURES = [
    "age_group", "gender", "race", "admission_type", "discharge_disposition",
    "insulin", "hba1c_result", "diabetesMed",
]
TARGET = "readmitted_30d"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found — run `python generate_sample_data.py` first."
        )
    return pd.read_csv(DATA_PATH)


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def get_candidate_models() -> dict:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
        ),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    logger.info("Loading data from %s", DATA_PATH)
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    logger.info("Class balance: %.1f%% readmitted (%d / %d)", y.mean() * 100, y.sum(), len(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    # Fit preprocessor once, transform both splits, then SMOTE the training set only
    # (SMOTE must never see the test set — that would leak information).
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_enc, y_train)
    logger.info(
        "SMOTE: %d -> %d training rows (balanced %.0f%% / %.0f%%)",
        len(y_train), len(y_train_bal), (1 - y_train_bal.mean()) * 100, y_train_bal.mean() * 100
    )

    results = []
    fitted_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in get_candidate_models().items():
        model.fit(X_train_bal, y_train_bal)
        preds = model.predict(X_test_enc)
        probs = model.predict_proba(X_test_enc)[:, 1]

        cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring="roc_auc")

        metrics = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1_score": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4),
            "cv_auc_mean": round(cv_scores.mean(), 4),
            "cv_auc_std": round(cv_scores.std(), 4),
        }
        results.append(metrics)
        fitted_models[name] = model
        logger.info("%s | test ROC-AUC %.4f | cv ROC-AUC %.4f ± %.4f",
                    name, metrics["roc_auc"], metrics["cv_auc_mean"], metrics["cv_auc_std"])

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    logger.info("Best model: %s (test ROC-AUC %.4f)", best_name, results_df.iloc[0]["roc_auc"])

    # ---- Confusion matrix + ROC curve data for the winning model ----
    best_preds = best_model.predict(X_test_enc)
    best_probs = best_model.predict_proba(X_test_enc)[:, 1]
    cm = confusion_matrix(y_test, best_preds)
    pd.DataFrame(
        cm, index=["actual_no", "actual_yes"], columns=["pred_no", "pred_yes"]
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    fpr, tpr, _ = roc_curve(y_test, best_probs)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(OUTPUT_DIR / "roc_curve.csv", index=False)

    # ---- SHAP feature importance (best model) ----
    logger.info("Computing SHAP values for %s ...", best_name)
    if best_name in ("random_forest", "xgboost", "decision_tree"):
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_enc)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs_shap = pd.Series(abs(sv).mean(axis=0), index=feature_names)
    else:
        explainer = shap.LinearExplainer(best_model, X_train_bal)
        shap_values = explainer.shap_values(X_test_enc)
        mean_abs_shap = pd.Series(abs(shap_values).mean(axis=0), index=feature_names)

    shap_importance = (
        mean_abs_shap.sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", 0: "mean_abs_shap"})
    )
    shap_importance.columns = ["feature", "mean_abs_shap"]
    shap_importance.to_csv(OUTPUT_DIR / "feature_importance_shap.csv", index=False)

    # ---- Risk scoring on full dataset ----
    X_full_enc = preprocessor.transform(X)
    full_probs = best_model.predict_proba(X_full_enc)[:, 1]
    scored = df.copy()
    scored["risk_score"] = full_probs
    scored["risk_tier"] = pd.cut(
        scored["risk_score"], bins=[0, 0.25, 0.60, 1],
        labels=["Low", "Moderate", "High"], include_lowest=True,
    )
    scored.to_csv(OUTPUT_DIR / "scored_patients.csv", index=False)

    tier_summary = scored["risk_tier"].value_counts(normalize=True).round(4) * 100
    logger.info("Risk tier distribution: %s", tier_summary.to_dict())

    summary = {
        "best_model": best_name,
        "test_roc_auc": float(results_df.iloc[0]["roc_auc"]),
        "test_accuracy": float(results_df.iloc[0]["accuracy"]),
        "cv_roc_auc_mean": float(results_df.iloc[0]["cv_auc_mean"]),
        "cv_roc_auc_std": float(results_df.iloc[0]["cv_auc_std"]),
        "readmit_rate": round(float(y.mean()), 4),
        "n_patients": int(len(df)),
        "n_test": int(len(y_test)),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    joblib.dump(best_model, MODEL_DIR / "readmission_model.joblib")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    print("\n" + "=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)
    print(f"Best model: {best_name}  |  Test ROC-AUC: {summary['test_roc_auc']}")
    print(f"Outputs written to: {OUTPUT_DIR}/")
    print(f"Model saved to: {MODEL_DIR}/readmission_model.joblib")


if __name__ == "__main__":
    main()
