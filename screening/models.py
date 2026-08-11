from django.db import models


class JobPosting(models.Model):
    """A job description that resumes will be screened against."""
    title = models.CharField(max_length=200)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title


class Candidate(models.Model):
    """A candidate resume submitted for a given job posting."""
    job_posting = models.ForeignKey(
        JobPosting, on_delete=models.CASCADE, related_name="candidates"
    )
    name = models.CharField(max_length=200)
    resume_text = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return self.name


class ScreeningResult(models.Model):
    """
    Output of the trained ML classifier for one candidate against one
    job posting. Stores everything needed to render the ranked results
    page without re-running the model.
    """
    LABEL_CHOICES = [
        ("Shortlist", "Shortlist"),
        ("Reject", "Reject"),
    ]

    job_posting = models.ForeignKey(
        JobPosting, on_delete=models.CASCADE, related_name="results"
    )
    candidate = models.ForeignKey(
        Candidate, on_delete=models.CASCADE, related_name="results"
    )

    shortlist_probability = models.FloatField()
    predicted_label = models.CharField(max_length=20, choices=LABEL_CHOICES)
    tfidf_similarity = models.FloatField()
    skill_overlap = models.FloatField()
    matched_skills = models.JSONField(default=list)
    resume_years = models.FloatField(default=0)
    jd_years_required = models.FloatField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-shortlist_probability"]

    def __str__(self):
        return f"{self.candidate.name} -> {self.job_posting.title} ({self.shortlist_probability}%)"

    @property
    def matched_skills_display(self):
        return ", ".join(self.matched_skills) if self.matched_skills else "-"
