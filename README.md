# AI Resume Screening and Candidate Ranking System (Django MVT + Trained ML Model)

A full-stack web application that screens and ranks candidate resumes
against a job description using a **trained machine learning classifier**
(not just similarity scoring), built with **Django (Model-View-Template)**
on the backend/frontend and **scikit-learn** for the ML layer.

## What Makes This "Trained ML" (Not Just Similarity Matching)

Unlike a simple TF-IDF/cosine-similarity matcher, this project actually
**trains a supervised classifier** to predict *Shortlist* vs *Reject*:

1. **Synthetic labeled dataset** (`screening/ml/generate_training_data.py`)
   — generates 1,500 realistic (job description, resume, label) triples
   across 5 job categories (Data Scientist, Software Engineer, Marketing
   Specialist, Sales Executive, HR Manager), with labels simulating a
   recruiter's imperfect, noisy shortlist decision (not a hard rule).
2. **Feature engineering** (`screening/ml/features.py`) — TF-IDF cosine
   similarity, skill-keyword overlap ratio, years-of-experience extraction,
   experience gap, and resume length.
3. **Model training & comparison** (`screening/ml/train_model.py`) — trains
   both Logistic Regression and Random Forest, evaluates on a held-out
   test split (accuracy/precision/recall/F1), and saves the best-performing
   model.
   - **Achieved: ~94% accuracy, ~0.92 F1** on held-out test data
     (Random Forest selected as the best model).
4. **Inference** (`screening/ml/predictor.py`) — loads the trained model
   and scores new resume/JD pairs in the Django app in real time.

## Architecture (Django MVT)

| MVT Layer | Files |
|---|---|
| **Model** | `screening/models.py` — `JobPosting`, `Candidate`, `ScreeningResult` |
| **View** | `screening/views.py` — handles form submission, runs the trained model, persists results |
| **Template** | `screening/templates/screening/*.html` — `home.html`, `results.html`, `history.html`, `about_model.html` |

```
django_resume_screening/
├── manage.py
├── requirements.txt
├── resume_screening/          # Django project settings/urls
│   ├── settings.py
│   ├── urls.py
├── screening/                 # Django app
│   ├── models.py              # JobPosting, Candidate, ScreeningResult
│   ├── views.py                # home, results, history, about_model
│   ├── forms.py                 # JobPostingForm
│   ├── urls.py
│   ├── admin.py
│   ├── templates/screening/
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── results.html
│   │   ├── history.html
│   │   └── about_model.html
│   ├── static/screening/css/style.css
│   └── ml/                     # ML training + inference (independent of Django)
│       ├── generate_training_data.py
│       ├── features.py
│       ├── train_model.py
│       ├── predictor.py
│       ├── data/training_data.csv
│       └── models/              # trained model artifacts (.joblib)
│           ├── classifier.joblib
│           ├── tfidf_vectorizer.joblib
│           ├── feature_scaler.joblib
│           └── metadata.joblib
└── sample_data/                 # Try-it-now job description + resumes
```

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Already done, but to retrain from scratch) Generate data & train the model
```bash
python screening/ml/generate_training_data.py   # creates screening/ml/data/training_data.csv
python screening/ml/train_model.py               # trains + saves model to screening/ml/models/
```
This prints accuracy/precision/recall/F1 for both Logistic Regression and
Random Forest, and saves the best one.

### 3. Set up the database
```bash
python manage.py migrate
```

### 4. (Optional) Create an admin user
```bash
python manage.py createsuperuser
```

### 5. Run the development server
```bash
python manage.py runserver
```
Visit **http://127.0.0.1:8000/**

### 6. Try it out
- Go to the home page, paste a job description (or use
  `sample_data/jd_data_scientist.txt`)
- Upload the sample resumes in `sample_data/` (or paste resume text
  manually) — you can add as many candidates as you like
- Click **Screen & Rank Candidates**
- View the ranked results page with each candidate's:
  - Predicted label (Shortlist / Reject) from the trained classifier
  - Shortlist probability (%)
  - TF-IDF text similarity (%)
  - Skill overlap (%) and matched skills
  - Years of experience vs. years required
- Visit **History** to see all past job postings and their results
- Visit **Model Info** to see the trained model's metrics and features

## Pages / Routes

| URL | View | Purpose |
|---|---|---|
| `/` | `home` | Create job posting, upload/paste resumes, run screening |
| `/job/<id>/results/` | `results` | Ranked results for one job posting |
| `/history/` | `history` | List of all past job postings |
| `/about-model/` | `about_model` | Trained model metrics & feature list |
| `/admin/` | Django admin | Browse/edit JobPosting, Candidate, ScreeningResult records |

## Retraining / Extending the Model

- **More/better training data**: edit `JOB_CATEGORIES` in
  `generate_training_data.py` to add job categories or skills, or replace
  the synthetic generator entirely with real labeled historical hiring
  data if you have it (same CSV schema: `job_category, job_description,
  resume_text, label`).
- **Different model**: add another candidate to the `candidates` dict in
  `train_model.py` (e.g. `GradientBoostingClassifier`, `SVC`) — it will be
  compared automatically and saved if it wins on F1 score.
- **New features**: add functions to `features.py` and include them in
  `build_feature_row()` / `FEATURE_NAMES` — training and inference stay
  in sync automatically since both import from this shared module.
- **Real resume files (PDF/DOCX)**: add a text-extraction step before
  candidates reach the view (e.g. `PyPDF2`, `python-docx`), then pass the
  extracted text into the existing pipeline unchanged.

## Tech Stack

- **Django** — MVT web framework (models, views, templates, forms, admin)
- **scikit-learn** — TF-IDF vectorization, Logistic Regression, Random
  Forest, train/test split, evaluation metrics
- **joblib** — persisting trained model artifacts
- **SQLite** (default Django DB) — stores job postings, candidates, and
  screening results
