"""
features.py
------------
Feature engineering shared by both the TRAINING script and the Django
inference layer, so the model always sees data prepared the same way.

Features used by the classifier:
    1. tfidf_cosine_similarity  - semantic/textual closeness of resume to JD
    2. skill_overlap_ratio      - fraction of JD's required skills found in resume
    3. resume_years_experience  - years of experience extracted from resume text
    4. jd_years_required        - years of experience required, extracted from JD
    5. experience_gap           - resume_years - jd_years (can be negative)
    6. resume_length            - word count of resume (proxy for detail/effort)
"""

import re

import numpy as np

# Keyword taxonomy used for skill-overlap scoring (kept in sync with the
# skills used in generate_training_data.py, but works generally for any
# resume/JD text).
SKILL_KEYWORDS = [
    # Core languages
    "python", "java", "javascript", "typescript", "c++", "c#", "php",
    "ruby", "go", "golang", "swift", "kotlin", "scala",
    # Web / full-stack frameworks & tools
    "django", "flask", "fastapi", "react", "angular", "vue", "next.js",
    "node", "express", "spring", "spring boot", ".net", "rails",
    "laravel", "html", "css", "sass", "tailwind", "bootstrap", "webpack",
    "rest api", "graphql", "microservices", "oauth",
    # Databases
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "sqlite", "oracle",
    # DevOps / cloud
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins",
    "terraform", "linux", "git", "github actions",
    # Data / ML
    "machine learning", "deep learning", "nlp", "data analysis",
    "pandas", "numpy", "tensorflow", "pytorch", "statistics",
    "tableau", "power bi", "excel",
    # Testing & process
    "unit testing", "agile", "scrum", "project management",
    # Soft skills
    "communication", "leadership", "creativity",
    # Marketing
    "seo", "content marketing", "social media", "google analytics",
    "email marketing", "copywriting", "branding", "market research",
    # Sales
    "crm", "salesforce", "negotiation", "lead generation",
    "cold calling", "account management", "business development",
    "relationship building",
    # HR
    "recruitment", "onboarding", "employee relations", "payroll",
    "performance management", "hr policies", "compliance",
    "conflict resolution", "empathy",
]

# Skills containing characters that aren't "word" characters in regex
# (e.g. c++, .net, next.js, ci/cd) need custom boundary handling instead
# of the standard \b word-boundary approach.
_SKILL_PATTERNS = {}
for _skill in SKILL_KEYWORDS:
    _escaped = re.escape(_skill)
    _SKILL_PATTERNS[_skill] = re.compile(rf"(?<![a-z0-9]){_escaped}(?![a-z0-9])")


def extract_years_experience(text: str) -> float:
    """Pull a 'years of experience' number out of free text, defaulting to 0."""
    text = text.lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", text)
    if matches:
        return max(float(m) for m in matches)
    return 0.0


def extract_skills(text: str) -> set:
    """
    Find which known skills appear in `text`, using boundary-aware matching
    so short skill names don't false-positive inside longer words
    (e.g. 'sql' no longer matches inside 'postgresql', 'java' no longer
    matches inside 'javascript', 'excel' no longer matches inside
    'excellent').
    """
    text_lower = text.lower()
    return {skill for skill, pattern in _SKILL_PATTERNS.items() if pattern.search(text_lower)}


def skill_overlap_ratio(resume_text: str, jd_text: str) -> float:
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    if not jd_skills:
        return 0.0
    return len(jd_skills & resume_skills) / len(jd_skills)


def build_feature_row(resume_text: str, jd_text: str, tfidf_cosine_sim: float) -> list:
    """Assemble the numeric feature vector for one (resume, jd) pair.
    `tfidf_cosine_sim` is computed separately (batched) via the fitted
    TfidfVectorizer + cosine_similarity, then passed in here.
    """
    resume_years = extract_years_experience(resume_text)
    jd_years = extract_years_experience(jd_text)
    overlap = skill_overlap_ratio(resume_text, jd_text)
    experience_gap = resume_years - jd_years
    resume_length = len(resume_text.split())

    return [
        tfidf_cosine_sim,
        overlap,
        resume_years,
        jd_years,
        experience_gap,
        resume_length,
    ]


FEATURE_NAMES = [
    "tfidf_cosine_similarity",
    "skill_overlap_ratio",
    "resume_years_experience",
    "jd_years_required",
    "experience_gap",
    "resume_length",
]
