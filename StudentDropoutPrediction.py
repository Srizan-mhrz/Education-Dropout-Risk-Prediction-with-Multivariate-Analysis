"""
=============================================================================
Student Dropout Risk Prediction System
=============================================================================
Dataset : "Predict Students' Dropout and Academic Success"
          UCI Machine Learning Repository

Model   : Logistic Regression
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0. IMPORTS
# ---------------------------------------------------------------------------

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



# ---------------------------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------------------------

def load_and_preprocess(filepath="dataset.csv"):

    print("=" * 60)
    print("STEP 1 — Loading & Preprocessing Data")
    print("=" * 60)

    df = pd.read_csv(filepath, sep=None, engine="python")

    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    if "Target" not in df.columns:
        raise ValueError("ERROR: 'Target' column not found.")

    print("\nOriginal target distribution:")
    print(df["Target"].value_counts())

    # Binary target
    df["Target_Binary"] = (df["Target"] == "Dropout").astype(int)
    df.drop(columns=["Target"], inplace=True)

    X = df.drop(columns=["Target_Binary"])
    y = df["Target_Binary"]
    feature_names = X.columns.tolist()

    # Encode non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        le = LabelEncoder()
        for col in non_numeric:
            X[col] = le.fit_transform(X[col].astype(str))

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE (training data only)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    print("\nAfter SMOTE class balance:")
    print(pd.Series(y_train_res).value_counts())

    # ✅ SMOTE VISUALIZATIONS
    plot_smote_before_after(y_train, y_train_res)
    plot_smote_pca(X_train_scaled, y_train, X_train_res, y_train_res)

    print("\nPreprocessing complete.\n")

    return X_train_res, X_test_scaled, y_train_res, y_test, feature_names, scaler


# ---------------------------------------------------------------------------
# 2. MODEL TRAINING
# ---------------------------------------------------------------------------

def train_model(X_train, y_train):

    print("=" * 60)
    print("STEP 2 — Model Training & Cross-Validation")
    print("=" * 60)

    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        C=0.5,
        solver="lbfgs",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results = cross_validate(
        lr_model, X_train, y_train,
        cv=cv, scoring=scoring, n_jobs=-1
    )

    print("\n5-Fold CV Results:")
    for metric in scoring:
        vals = cv_results[f"test_{metric}"]
        print(f"{metric:12s}: {vals.mean():.4f} +/- {vals.std():.4f}")

    lr_model.fit(X_train, y_train)

    return lr_model


# ---------------------------------------------------------------------------
# 3. EVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(lr_model, X_test, y_test):

    y_pred = lr_model.predict(X_test)
    y_proba = lr_model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Non-Dropout", "Dropout"]))

    auc_score = roc_auc_score(y_test, y_proba)

    os.makedirs("outputs", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm,
                           display_labels=["Non-Dropout", "Dropout"]).plot()
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_auc_curve.png")
    plt.close()


# ---------------------------------------------------------------------------
# 4. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def plot_feature_importance(lr_model, feature_names, top_n=10):

    os.makedirs("outputs", exist_ok=True)

    coefs = lr_model.coef_[0]
    idx = np.argsort(np.abs(coefs))[::-1][:top_n]

    plt.barh(
        [feature_names[i] for i in idx][::-1],
        coefs[idx][::-1]
    )
    plt.axvline(0)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main(dataset_path="dataset.csv"):

    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess(dataset_path)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    plot_feature_importance(model, feature_names)

    print("\nPipeline complete!")
    print("All outputs saved inside /outputs")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset.csv")
    args = parser.parse_args()

    main(dataset_path=args.data)