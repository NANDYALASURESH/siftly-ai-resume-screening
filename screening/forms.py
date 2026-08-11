from django import forms

from .models import JobPosting


class JobPostingForm(forms.ModelForm):
    class Meta:
        model = JobPosting
        fields = ["title", "description"]
        widgets = {
            "title": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "e.g. Data Scientist",
            }),
            "description": forms.Textarea(attrs={
                "class": "form-control",
                "rows": 8,
                "placeholder": "Paste the full job description here...",
            }),
        }


class ManualResumeForm(forms.Form):
    """Fallback: paste a single candidate's resume manually."""
    candidate_name = forms.CharField(
        max_length=200, required=False,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Candidate name"}),
    )
    resume_text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"class": "form-control", "rows": 6,
                                      "placeholder": "Paste resume text..."}),
    )
