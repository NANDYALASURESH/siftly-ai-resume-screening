from django.contrib import admin

from .models import JobPosting, Candidate, ScreeningResult


@admin.register(JobPosting)
class JobPostingAdmin(admin.ModelAdmin):
    list_display = ("title", "created_at")
    search_fields = ("title", "description")


@admin.register(Candidate)
class CandidateAdmin(admin.ModelAdmin):
    list_display = ("name", "job_posting", "uploaded_at")
    search_fields = ("name",)


@admin.register(ScreeningResult)
class ScreeningResultAdmin(admin.ModelAdmin):
    list_display = ("candidate", "job_posting", "predicted_label", "shortlist_probability")
    list_filter = ("predicted_label",)
