from django.urls import path
from django.contrib.auth import views as auth_views

from . import views

app_name = "screening"

urlpatterns = [
    path("", views.home, name="home"),
    path("job/<int:job_id>/results/", views.results, name="results"),
    path("history/", views.history, name="history"),
    path("about-model/", views.about_model, name="about_model"),

    path("signup/", views.signup, name="signup"),
    path("login/", auth_views.LoginView.as_view(
        template_name="screening/login.html",
        redirect_authenticated_user=True,
    ), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
]
