"""
predictor.py
-------------
INFERENCE layer used by the Django app. Loads the trained model artifacts
(classifier, TF-IDF vectorizer, feature scaler) once and exposes a simple
`screen_resume()` function that returns a shortlist probability + predicted
label + supporting scores for a single (resume_text, jd_text) pair.
"""

import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .features import build_feature_row, extract_skills, FEATURE_NAMES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

_classifier = None
_vectorizer = None
_scaler = None
_metadata = None


def _load_artifacts():
    global _classifier, _vectorizer, _scaler, _metadata
    if _classifier is None:
        _classifier = joblib.load(os.path.join(MODEL_DIR, "classifier.joblib"))
        _vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
        _scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.joblib"))
        _metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))
    return _classifier, _vectorizer, _scaler, _metadata


def model_is_trained() -> bool:
    """Check whether trained model artifacts exist on disk."""
    required = ["classifier.joblib", "tfidf_vectorizer.joblib",
                "feature_scaler.joblib", "metadata.joblib"]
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required)


def get_model_metadata() -> dict:
    _, _, _, metadata = _load_artifacts()
    return metadata


def screen_resume(resume_text: str, jd_text: str) -> dict:
    """
    Score a single resume against a job description using the trained
    classifier.

    Returns a dict with:
        shortlist_probability : float (0-100)
        predicted_label       : "Shortlist" | "Reject"
        tfidf_similarity      : float (0-100)
        skill_overlap         : float (0-100)
        matched_skills        : list[str]
        resume_years          : float
        jd_years_required     : float
    """
    classifier, vectorizer, scaler, _ = _load_artifacts()

    resume_vec = vectorizer.transform([resume_text])
    jd_vec = vectorizer.transform([jd_text])
    cos_sim = float(cosine_similarity(resume_vec, jd_vec)[0][0])

    feature_row = build_feature_row(resume_text, jd_text, cos_sim)
    X = np.array([feature_row])
    X_scaled = scaler.transform(X)

    proba = classifier.predict_proba(X_scaled)[0]
    shortlist_proba = float(proba[1])  # class 1 = shortlist
    predicted_label = "Shortlist" if shortlist_proba >= 0.5 else "Reject"

    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    matched = sorted(jd_skills & resume_skills)
    overlap_ratio = (len(matched) / len(jd_skills)) if jd_skills else 0.0

    return {
        "shortlist_probability": round(shortlist_proba * 100, 2),
        "predicted_label": predicted_label,
        "tfidf_similarity": round(cos_sim * 100, 2),
        "skill_overlap": round(overlap_ratio * 100, 2),
        "matched_skills": matched,
        "resume_years": feature_row[FEATURE_NAMES.index("resume_years_experience")],
        "jd_years_required": feature_row[FEATURE_NAMES.index("jd_years_required")],
    }


def rank_resumes(resumes: dict, jd_text: str) -> list:
    """
    Score and rank multiple candidates against one job description.
    `resumes` is {candidate_name: resume_text}.
    Returns a list of dicts sorted by shortlist_probability descending.
    """
    results = []
    for name, text in resumes.items():
        scores = screen_resume(text, jd_text)
        scores["candidate_name"] = name
        results.append(scores)
    results.sort(key=lambda r: r["shortlist_probability"], reverse=True)
    return results
