"""
train_model.py
----------------
TRAINS the resume-shortlisting classifier.

Pipeline:
    1. Load the synthetic labeled dataset (job_description, resume_text, label)
    2. Fit a TF-IDF vectorizer jointly over resumes + job descriptions
    3. Compute per-pair cosine similarity + engineered features (features.py)
    4. Train and compare Logistic Regression and Random Forest classifiers
    5. Evaluate on a held-out test split (accuracy, precision, recall, F1)
    6. Save the best model + TF-IDF vectorizer + feature scaler to
       screening/ml/models/ for use by the Django app at inference time

Run:
    python3 screening/ml/train_model.py
"""

import os
import sys
import csv
import joblib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

sys.path.append(os.path.dirname(__file__))
from features import build_feature_row, FEATURE_NAMES  # noqa: E402

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_dataset(path):
    resumes, jds, labels = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            resumes.append(row["resume_text"])
            jds.append(row["job_description"])
            labels.append(int(row["label"]))
    return resumes, jds, np.array(labels)


def main():
    print("Loading dataset...")
    resumes, jds, y = load_dataset(DATA_PATH)
    print(f"  {len(resumes)} labeled resume/JD pairs loaded")

    # ------------------------------------------------------------
    # 1. Fit TF-IDF over the combined corpus of resumes + JDs
    # ------------------------------------------------------------
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=2000, ngram_range=(1, 2)
    )
    all_docs = resumes + jds
    vectorizer.fit(all_docs)

    resume_vecs = vectorizer.transform(resumes)
    jd_vecs = vectorizer.transform(jds)

    # Row-wise cosine similarity between matching resume/JD pairs
    print("Computing TF-IDF cosine similarities...")
    cos_sims = np.array([
        cosine_similarity(resume_vecs[i], jd_vecs[i])[0][0]
        for i in range(len(resumes))
    ])

    # ------------------------------------------------------------
    # 2. Build the engineered feature matrix
    # ------------------------------------------------------------
    print("Building engineered features...")
    X = np.array([
        build_feature_row(resumes[i], jds[i], cos_sims[i])
        for i in range(len(resumes))
    ])

    # ------------------------------------------------------------
    # 3. Train / test split + scaling
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------
    # 4. Train candidate models and compare
    # ------------------------------------------------------------
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        results[name] = {
            "model": model, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1,
        }
        print(f"\n=== {name} ===")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1 score:  {f1:.3f}")
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds, target_names=["Reject", "Shortlist"]))

    # ------------------------------------------------------------
    # 5. Pick the best model by F1 score and persist everything needed
    #    for inference (model, vectorizer, scaler, feature names)
    # ------------------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} (F1={results[best_name]['f1']:.3f})")

    if best_name == "logistic_regression":
        coefs = dict(zip(FEATURE_NAMES, best_model.coef_[0]))
        print("\nFeature importance (logistic regression coefficients):")
        for feat, coef in sorted(coefs.items(), key=lambda x: -abs(x[1])):
            print(f"  {feat:28s} {coef:+.3f}")
    else:
        importances = dict(zip(FEATURE_NAMES, best_model.feature_importances_))
        print("\nFeature importance (random forest):")
        for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {feat:28s} {imp:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "classifier.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "feature_scaler.joblib"))

    metadata = {
        "model_name": best_name,
        "feature_names": FEATURE_NAMES,
        "metrics": {k: v for k, v in results[best_name].items() if k != "model"},
        "n_training_samples": len(X_train),
        "n_test_samples": len(X_test),
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, "metadata.joblib"))

    print(f"\nSaved model artifacts to {MODEL_DIR}/")
    print("  - classifier.joblib")
    print("  - tfidf_vectorizer.joblib")
    print("  - feature_scaler.joblib")
    print("  - metadata.joblib")


if __name__ == "__main__":
    main()
