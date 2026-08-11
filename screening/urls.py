from django.urls import path

from . import views

app_name = "screening"

urlpatterns = [
    path("", views.home, name="home"),
    path("job/<int:job_id>/results/", views.results, name="results"),
    path("history/", views.history, name="history"),
    path("about-model/", views.about_model, name="about_model"),
]
