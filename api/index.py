"""
Vercel serverless entrypoint.

Vercel's Python runtime looks for a WSGI-compatible `app` (or `handler`)
callable in api/index.py and routes every request in vercel.json to it.
This just hands off to Django's normal WSGI application.
"""
import os
import sys
from pathlib import Path

# Make the Django project importable (repo root, one level up from /api).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'resume_screening.settings')

from django.core.wsgi import get_wsgi_application  # noqa: E402

app = get_wsgi_application()
