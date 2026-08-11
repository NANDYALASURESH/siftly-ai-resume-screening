# Deploying this project

The project is now configured for both **Render** and **Vercel**. Read the
"Which one should I use" note first — they are not equally good fits for a
Django app like this one.

Files added/changed for deployment:
- `resume_screening/settings.py` — reads `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`,
  `DATABASE_URL` from environment variables; adds WhiteNoise for static files.
- `requirements.txt` — added `gunicorn`, `whitenoise`, `dj-database-url`, `psycopg2-binary`.
- `build.sh`, `render.yaml` — Render build/start config.
- `vercel.json`, `api/index.py` — Vercel serverless entrypoint.
- `.env.example` — template for local env vars.

---

## Which one should I use?

**Render is the better fit and is what I'd recommend.** This is a standard
stateful Django app — it needs to run migrations, use a real database, and
serve requests from a long-running process. Render supports all of that
natively (free Postgres, persistent web service, a build step that runs
`collectstatic`/`migrate`).

**Vercel can technically host it, with real trade‑offs:**
- Vercel functions are stateless serverless functions with an ephemeral,
  mostly read-only filesystem — SQLite will not work, so you *must* point
  `DATABASE_URL` at an external Postgres (e.g. a free Render/Neon/Supabase
  database).
- There's no build hook to run `manage.py collectstatic` automatically, so
  you must generate `staticfiles/` locally and commit it (steps below).
- Each cold start reinitializes the Python process and reloads the ~4 MB ML
  model from disk, adding latency the first request after idle.
- No background workers/cron — irrelevant here since this app is
  request/response only, but worth knowing generally.

If you just want "it deployed and working," go with Render and skip the
Vercel section.

---

## Option A: Deploy to Render (recommended)

1. **Push this project to a GitHub repo** (Render deploys from Git).
2. Go to [render.com](https://render.com) → **New** → **Blueprint**, and
   point it at your repo. Render will detect `render.yaml` and provision:
   - a free Postgres database (`resume-screening-db`)
   - a web service (`resume-screening`) that runs `build.sh` then starts
     `gunicorn resume_screening.wsgi:application`
   - a random `SECRET_KEY` (auto-generated) and `DATABASE_URL` (wired to the
     database) automatically — you don't need to set these by hand.
3. Click **Apply**. First deploy takes a few minutes (installs
   scikit-learn/numpy, runs migrations, collects static files).
4. Once live, your app is at `https://resume-screening-<random>.onrender.com`
   (or whatever name you gave the service). `settings.py` already trusts
   Render's hostname automatically via `RENDER_EXTERNAL_HOSTNAME`.
5. **Create an admin user** (for `/admin/`): in the Render dashboard, open
   the web service → **Shell**, then run:
   ```
   python manage.py createsuperuser
   ```

No blueprint? You can instead click **New → Web Service**, connect the repo,
and manually set:
- Build command: `./build.sh`
- Start command: `gunicorn resume_screening.wsgi:application`
- Add a Postgres instance and copy its **Internal Database URL** into an
  env var named `DATABASE_URL`
- Add env var `SECRET_KEY` (any long random string)

**Note on the free plan:** Render's free web services spin down after 15
minutes of inactivity and take ~30-50s to wake back up on the next request.
Free Postgres databases also expire after 90 days unless upgraded.

---

## Option B: Deploy to Vercel

1. **Provision an external Postgres database first** (Vercel has no
   built-in DB for Python projects) — e.g. a free instance on
   [Neon](https://neon.tech) or [Render](https://render.com). Copy its
   connection string (`postgres://...`).

2. **Generate static files locally and commit them**, since Vercel won't
   run `manage.py` for you:
   ```bash
   pip install -r requirements.txt
   python manage.py collectstatic --no-input
   ```
   Then remove the `staticfiles/` line from `.gitignore` and commit the
   generated `staticfiles/` folder.

3. **Run migrations against the Postgres DB from your machine** (Vercel
   can't run one-off management commands either):
   ```bash
   export DATABASE_URL="postgres://...your Neon/Render URL..."
   python manage.py migrate
   python manage.py createsuperuser   # optional, for /admin/
   ```

4. **Push to GitHub**, then in the [Vercel dashboard](https://vercel.com):
   **Add New → Project → Import** your repo. Vercel will detect
   `vercel.json` and use the Python runtime automatically.

5. **Set environment variables** in Vercel project settings:
   - `SECRET_KEY` — any long random string
   - `DEBUG` — `False`
   - `DATABASE_URL` — the same Postgres URL from step 1
   - (leave `ALLOWED_HOSTS`/`VERCEL_URL` alone — `settings.py` reads
     Vercel's own `VERCEL_URL` env var automatically)

6. Click **Deploy**. Your app will be live at `https://<project>.vercel.app`.

If you ever change static assets (CSS etc.), repeat step 2 and redeploy.

---

## Local development (either way)

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env      # then edit values if needed
python manage.py migrate
python manage.py runserver
```

By default (no env vars set) it runs with `DEBUG` off but talking to a local
SQLite DB — for normal local dev, set `DEBUG=True` in `.env` too.
