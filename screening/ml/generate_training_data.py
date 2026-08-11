"""
generate_training_data.py
--------------------------
Generates a synthetic, labeled dataset of (job_description, resume, label)
pairs used to TRAIN a supervised "Shortlist / Reject" classifier.

Why synthetic data?
    Real hiring decisions (labeled resume/JD pairs) are usually private and
    not available for a demo project. This script simulates realistic
    resume/JD combinations across several job categories, with a
    probabilistic labeling rule (skill overlap + experience fit + noise)
    that mimics how a recruiter might actually shortlist candidates -
    imperfectly and with some randomness, not a hard cutoff.

Output:
    screening/ml/data/training_data.csv
    Columns: job_description, resume_text, job_category, label (1=shortlist, 0=reject)
"""

import random
import csv
import os

random.seed(42)

# --------------------------------------------------------------------
# Job categories, each with a pool of relevant skills and a JD template
# --------------------------------------------------------------------
JOB_CATEGORIES = {
    "Data Scientist": {
        "core_skills": ["python", "machine learning", "deep learning", "nlp",
                         "pandas", "numpy", "tensorflow", "pytorch",
                         "data analysis", "sql", "statistics"],
        "soft_skills": ["communication", "leadership", "project management",
                         "agile", "scrum"],
        "jd_template": (
            "We are hiring a Data Scientist. Required skills: {skills}. "
            "The candidate should have {years}+ years of experience "
            "building machine learning models and analyzing data."
        ),
    },
    "Software Engineer": {
        "core_skills": ["java", "python", "c++", "sql", "git", "docker",
                         "kubernetes", "react", "node", "aws", "linux"],
        "soft_skills": ["communication", "leadership", "agile", "scrum",
                         "project management"],
        "jd_template": (
            "We are hiring a Software Engineer. Required skills: {skills}. "
            "The candidate should have {years}+ years of backend or "
            "full-stack development experience."
        ),
    },
    "Full Stack Developer": {
        "core_skills": ["python", "django", "javascript", "react", "node",
                         "sql", "postgresql", "html", "css", "git",
                         "rest api", "docker", "aws"],
        "soft_skills": ["communication", "agile", "scrum",
                         "project management", "leadership"],
        "jd_template": (
            "We are hiring a Full Stack Developer. Required skills: "
            "{skills}. The candidate should have {years}+ years of "
            "professional experience building and shipping web "
            "applications."
        ),
    },
    "Frontend Developer": {
        "core_skills": ["javascript", "typescript", "react", "vue",
                         "angular", "html", "css", "sass", "tailwind",
                         "webpack", "git"],
        "soft_skills": ["communication", "creativity", "agile",
                         "project management"],
        "jd_template": (
            "We are hiring a Frontend Developer. Required skills: "
            "{skills}. The candidate should have {years}+ years of "
            "experience building user-facing web interfaces."
        ),
    },
    "Backend Developer": {
        "core_skills": ["python", "java", "django", "flask", "node",
                         "sql", "postgresql", "mongodb", "docker",
                         "kubernetes", "aws", "microservices", "rest api"],
        "soft_skills": ["communication", "leadership", "agile",
                         "project management"],
        "jd_template": (
            "We are hiring a Backend Developer. Required skills: "
            "{skills}. The candidate should have {years}+ years of "
            "experience designing and scaling server-side systems."
        ),
    },
    "DevOps Engineer": {
        "core_skills": ["docker", "kubernetes", "aws", "azure", "terraform",
                         "ci/cd", "jenkins", "linux", "git",
                         "github actions", "microservices"],
        "soft_skills": ["communication", "leadership", "agile",
                         "project management"],
        "jd_template": (
            "We are hiring a DevOps Engineer. Required skills: {skills}. "
            "The candidate should have {years}+ years of experience "
            "with infrastructure automation and cloud deployments."
        ),
    },
    "Marketing Specialist": {
        "core_skills": ["seo", "content marketing", "social media",
                         "google analytics", "email marketing",
                         "copywriting", "branding", "market research"],
        "soft_skills": ["communication", "creativity", "leadership",
                         "project management"],
        "jd_template": (
            "We are hiring a Marketing Specialist. Required skills: "
            "{skills}. The candidate should have {years}+ years of "
            "marketing campaign experience."
        ),
    },
    "Sales Executive": {
        "core_skills": ["crm", "salesforce", "negotiation", "lead generation",
                         "cold calling", "account management",
                         "business development"],
        "soft_skills": ["communication", "leadership", "negotiation",
                         "relationship building"],
        "jd_template": (
            "We are hiring a Sales Executive. Required skills: {skills}. "
            "The candidate should have {years}+ years of B2B sales "
            "experience."
        ),
    },
    "HR Manager": {
        "core_skills": ["recruitment", "onboarding", "employee relations",
                         "payroll", "performance management",
                         "hr policies", "compliance"],
        "soft_skills": ["communication", "leadership", "conflict resolution",
                         "empathy"],
        "jd_template": (
            "We are hiring an HR Manager. Required skills: {skills}. "
            "The candidate should have {years}+ years of human resources "
            "management experience."
        ),
    },
}

FILLER_PHRASES = [
    "Passionate about delivering high quality results.",
    "Proven track record of success in fast-paced environments.",
    "Strong problem solver with attention to detail.",
    "Excellent team player who enjoys collaborative work.",
    "Adaptable and eager to learn new tools and technologies.",
    "Consistently meets deadlines and exceeds expectations.",
]

NAMES_FIRST = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley",
               "Jamie", "Sam", "Priya", "Wei", "Fatima", "Diego", "Elena",
               "Noah", "Maya", "Liam", "Aisha", "Carlos", "Sofia", "Ravi"]
NAMES_LAST = ["Smith", "Johnson", "Lee", "Patel", "Garcia", "Kim", "Chen",
              "Brown", "Khan", "Silva", "Nguyen", "Kumar", "Rossi", "Muller"]


def random_name():
    return f"{random.choice(NAMES_FIRST)} {random.choice(NAMES_LAST)}"


def build_jd(category_name, category):
    skills_sample = random.sample(
        category["core_skills"], k=min(6, len(category["core_skills"]))
    )
    years_required = random.choice([2, 3, 4, 5])
    jd_text = category["jd_template"].format(
        skills=", ".join(skills_sample), years=years_required
    )
    return jd_text, skills_sample, years_required


def build_resume(category, match_strength):
    """
    match_strength in {"high", "medium", "low"} controls how well the
    synthetic candidate's resume aligns with the job category, simulating
    strong, partial, and weak-fit applicants.
    """
    all_skills = category["core_skills"] + category["soft_skills"]

    if match_strength == "high":
        n_skills = random.randint(6, 9)
        years_exp = random.randint(3, 8)
    elif match_strength == "medium":
        n_skills = random.randint(3, 5)
        years_exp = random.randint(1, 4)
    else:  # low
        n_skills = random.randint(0, 2)
        years_exp = random.randint(0, 2)

    n_skills = min(n_skills, len(all_skills))
    resume_skills = random.sample(all_skills, k=n_skills)

    # occasionally pull in 1-2 unrelated skills from another random category
    if random.random() < 0.3:
        other_cat = random.choice(
            [c for c in JOB_CATEGORIES.values() if c is not category]
        )
        resume_skills += random.sample(
            other_cat["core_skills"], k=min(2, len(other_cat["core_skills"]))
        )

    name = random_name()
    filler = " ".join(random.sample(FILLER_PHRASES, k=2))
    resume_text = (
        f"{name}. {years_exp} years of professional experience. "
        f"Skilled in {', '.join(resume_skills)}. {filler}"
    )
    return resume_text, resume_skills, years_exp


def label_pair(jd_skills, jd_years, resume_skills, resume_years):
    """
    Simulate a recruiter's shortlist decision:
    combines skill overlap ratio + experience fit, plus random noise,
    so the label isn't a perfectly deterministic function of the features
    (more realistic for training a classifier that generalizes).
    """
    jd_skill_set = set(jd_skills)
    resume_skill_set = set(resume_skills)
    overlap_ratio = (
        len(jd_skill_set & resume_skill_set) / len(jd_skill_set)
        if jd_skill_set else 0
    )
    experience_fit = 1.0 if resume_years >= jd_years else resume_years / max(jd_years, 1)

    score = 0.65 * overlap_ratio + 0.35 * experience_fit
    score += random.uniform(-0.12, 0.12)  # noise -> imperfect recruiter judgment

    return 1 if score >= 0.55 else 0


def generate_dataset(n_per_category=300):
    rows = []
    for category_name, category in JOB_CATEGORIES.items():
        for _ in range(n_per_category):
            jd_text, jd_skills, jd_years = build_jd(category_name, category)
            match_strength = random.choices(
                ["high", "medium", "low"], weights=[0.35, 0.35, 0.30]
            )[0]
            resume_text, resume_skills, resume_years = build_resume(
                category, match_strength
            )
            label = label_pair(jd_skills, jd_years, resume_skills, resume_years)
            rows.append({
                "job_category": category_name,
                "job_description": jd_text,
                "resume_text": resume_text,
                "label": label,
            })
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "training_data.csv")

    rows = generate_dataset(n_per_category=300)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["job_category", "job_description", "resume_text", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(r["label"] for r in rows)
    print(f"Generated {len(rows)} samples -> {out_path}")
    print(f"Positive (shortlist) rate: {n_pos / len(rows):.2%}")
