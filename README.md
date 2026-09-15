# Deploying the Employee Attrition Model to Render

This folder is a complete, self-contained API service — no MLflow tracking
server, no SQLite database, no absolute file paths. The trained model
(MLflow Registry Version 2, F1 = 0.5653, the best-performing registered
version) was exported once to `app/model/model.pkl`, so this runs identically
on any machine or cloud provider.

## What's in here

```
render_deploy/
├── app/
│   ├── main.py                  # FastAPI app: /, /health, /features, /predict
│   └── model/
│       ├── model.pkl             # the actual trained LogisticRegression model
│       └── feature_columns.json  # exact input column names + order
├── Dockerfile                    # builds the container Render will run
├── render.yaml                   # Render "Blueprint" — lets Render auto-configure the service
├── requirements.txt
└── sample_request.json           # a ready-to-use example payload for /predict
```

Already tested locally in a clean environment: `/predict` returns real
predictions and probabilities, `/health` returns 200, bad input returns a
clear 400 error instead of crashing.

## Step 1 — Put this folder in a GitHub repo

Render deploys from a Git repository, not a raw zip upload. If you don't
already have one:

1. Go to https://github.com/new, create a new repo (public or private).
2. On your machine, inside this unzipped folder:
   ```bash
   git init
   git add .
   git commit -m "Employee attrition prediction API"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

## Step 2 — Create the service on Render

1. Go to https://dashboard.render.com and sign in (free tier is fine).
2. Click **New +** → **Blueprint**.
3. Connect your GitHub account and select the repo you just pushed.
4. Render will detect `render.yaml` automatically and show the
   `employee-attrition-api` service — click **Apply**.
   - If you'd rather not use the Blueprint file, choose **New + → Web
     Service** instead, select the repo, set **Runtime: Docker**, and leave
     everything else as detected.
5. Click **Create Web Service**. Render will build the Docker image and
   deploy it — this takes 2-5 minutes on the free tier.

## Step 3 — Use your live API

Render gives you a public URL like:
```
https://employee-attrition-api.onrender.com
```

Test it:
```bash
curl https://employee-attrition-api.onrender.com/health

curl -X POST https://employee-attrition-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Interactive docs (Swagger UI) are automatically available at:
```
https://employee-attrition-api.onrender.com/docs
```

## Notes on the free tier

- Free Render web services **spin down after 15 minutes of inactivity** and
  take ~30-60 seconds to wake back up on the next request. This is normal,
  not a bug — upgrade to a paid instance to avoid it.
- No environment variables, secrets, or credentials are required — the
  model is baked into the image.

## Testing locally before you deploy (optional)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# in another terminal:
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

Or with Docker directly:
```bash
docker build -t attrition-api .
docker run -p 8000:8000 attrition-api
```
