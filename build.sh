#!/usr/bin/env bash
# Render build script. Render runs this automatically before starting the
# app (configured as the "Build Command" in render.yaml / dashboard).
set -o errexit

pip install -r requirements.txt

python manage.py collectstatic --no-input
python manage.py migrate
