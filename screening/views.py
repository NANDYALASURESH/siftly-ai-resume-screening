from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q
from django.shortcuts import render, redirect, get_object_or_404

from .forms import JobPostingForm, SignupForm
from .models import JobPosting, Candidate, ScreeningResult
from .ml.predictor import (
    screen_resume,
    model_is_trained,
    get_model_metadata,
)
from .utils import (
    extract_text_from_upload,
    extract_resumes_from_zip,
    UnsupportedFileType,
)


def signup(request):
    """Self-serve account creation. Logs the user in immediately after."""

    if request.user.is_authenticated:
        return redirect("screening:home")

    if request.method == "POST":
        form = SignupForm(request.POST)

        if form.is_valid():
            user = form.save()
            login(request, user)

            messages.success(
                request,
                f"Welcome, {user.username}! Your account is ready."
            )

            return redirect("screening:home")

    else:
        form = SignupForm()

    return render(
        request,
        "screening/signup.html",
        {"form": form},
    )


@login_required
def home(request):
    """
    Landing page: create a job posting, upload/paste resumes,
    and trigger the trained model to screen and rank them.
    """

    if request.method == "POST":

        job_form = JobPostingForm(request.POST)

        if not job_form.is_valid():
            messages.error(
                request,
                "Please provide a job title and description."
            )

            return render(
                request,
                "screening/home.html",
                {
                    "job_form": job_form,
                    "model_ready": model_is_trained(),
                },
            )

        # Gather resumes from uploaded PDF / Word files
        candidates_data = []

        uploaded_files = request.FILES.getlist("resumes")

        for f in uploaded_files:
            try:
                content = extract_text_from_upload(f)

            except UnsupportedFileType as exc:
                messages.error(request, str(exc))
                continue

            except (ValueError, ImportError) as exc:
                messages.error(
                    request,
                    f"Couldn't process '{f.name}': {exc}"
                )
                continue

            name = f.name.rsplit(".", 1)[0]

            if content.strip():
                candidates_data.append(
                    (name, content)
                )
            else:
                messages.warning(
                    request,
                    f"'{f.name}' had no extractable text and was skipped."
                )

        # Gather resumes from uploaded ZIP archive
        zip_file = request.FILES.get("resumes_zip")

        if zip_file:
            try:
                zip_candidates, skipped = extract_resumes_from_zip(
                    zip_file
                )

                candidates_data.extend(zip_candidates)

                if zip_candidates:
                    messages.success(
                        request,
                        f"Extracted {len(zip_candidates)} resume(s) "
                        f"from '{zip_file.name}'.",
                    )

                for skipped_name, reason in skipped:
                    messages.warning(
                        request,
                        f"Skipped '{skipped_name}' in ZIP: {reason}",
                    )

            except ValueError as exc:
                messages.error(request, str(exc))

        # Gather manually pasted resumes
        manual_names = request.POST.getlist("manual_name")
        manual_texts = request.POST.getlist("manual_resume")

        for name, text in zip(manual_names, manual_texts):

            if text.strip():
                candidates_data.append(
                    (
                        name.strip() or "Unnamed Candidate",
                        text,
                    )
                )

        # Make sure at least one resume exists
        if not candidates_data:

            messages.error(
                request,
                "Please upload or paste at least one resume."
            )

            return render(
                request,
                "screening/home.html",
                {
                    "job_form": job_form,
                    "model_ready": model_is_trained(),
                },
            )

        # Save job posting
        job_posting = job_form.save()

        # Create candidates and run ML model
        for name, resume_text in candidates_data:

            candidate = Candidate.objects.create(
                job_posting=job_posting,
                name=name,
                resume_text=resume_text,
            )

            scores = screen_resume(
                resume_text,
                job_posting.description,
            )

            ScreeningResult.objects.create(
                job_posting=job_posting,
                candidate=candidate,
                shortlist_probability=scores["shortlist_probability"],
                predicted_label=scores["predicted_label"],
                tfidf_similarity=scores["tfidf_similarity"],
                skill_overlap=scores["skill_overlap"],
                matched_skills=scores["matched_skills"],
                resume_years=scores["resume_years"],
                jd_years_required=scores["jd_years_required"],
            )

        return redirect(
            "screening:results",
            job_id=job_posting.id,
        )

    # GET request
    job_form = JobPostingForm()

    return render(
        request,
        "screening/home.html",
        {
            "job_form": job_form,
            "model_ready": model_is_trained(),
        },
    )


@login_required
def results(request, job_id):
    """Ranked screening results for a given job posting."""

    job_posting = get_object_or_404(
        JobPosting,
        id=job_id,
    )

    ranked_results = (
        job_posting.results
        .all()
        .select_related("candidate")
    )

    total_count = ranked_results.count()

    shortlisted_count = ranked_results.filter(
        predicted_label="Shortlist"
    ).count()

    reject_count = total_count - shortlisted_count

    if total_count:
        avg_match = (
            sum(
                r.shortlist_probability
                for r in ranked_results
            )
            / total_count
        )
    else:
        avg_match = 0

    return render(
        request,
        "screening/results.html",
        {
            "job_posting": job_posting,
            "results": ranked_results,
            "shortlisted_count": shortlisted_count,
            "reject_count": reject_count,
            "avg_match": round(avg_match, 1),
            "total_count": total_count,
        },
    )


@login_required
def history(request):
    """List of past job postings, most recent first."""

    job_postings = JobPosting.objects.all().annotate(
        total_screened=Count("results"),
        total_shortlisted=Count(
            "results",
            filter=Q(
                results__predicted_label="Shortlist"
            ),
        ),
    )

    return render(
        request,
        "screening/history.html",
        {
            "job_postings": job_postings,
        },
    )


@login_required
def about_model(request):
    """Displays metadata/metrics about the trained ML model."""

    ready = model_is_trained()

    metadata = (
        get_model_metadata()
        if ready
        else None
    )

    return render(
        request,
        "screening/about_model.html",
        {
            "model_ready": ready,
            "metadata": metadata,
        },
    )