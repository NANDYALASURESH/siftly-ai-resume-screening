from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

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


class SignupForm(UserCreationForm):
    """Basic username + password signup, styled to match the app."""
    email = forms.EmailField(
        required=False,
        widget=forms.EmailInput(attrs={"class": "form-control", "placeholder": "you@company.com"}),
    )

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].widget.attrs.update({
            "class": "form-control", "placeholder": "Choose a username", "autofocus": True,
        })
        self.fields["password1"].widget.attrs.update({
            "class": "form-control", "placeholder": "Create a password",
        })
        self.fields["password2"].widget.attrs.update({
            "class": "form-control", "placeholder": "Confirm password",
        })
