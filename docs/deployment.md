# Deployment, Docker, Kubernetes and CI

## Prediction API

The FastAPI service provides `GET /health` and `POST /predict`.

The prediction request contains eight fields: queue waiting
time, agent experience, previous calls, issue count, hour,
age, language, and team.

The model predicts `handle_min` in minutes.
The API returns the result as `predicted_handle_min`,
rounded to two decimal places.

## Input Validation

The API rejects `previous_calls` above 20 and
`agent_exp_months` below 0 with HTTP 422.
Unrecognized fields are also rejected.

## Docker

Our image is named `mp216-api:1.0`.

The image includes an exported Version 3 champion model
as `models/model.joblib`.

The API loads that file inside Docker, while local
MLflow-based runs can use the registry.

Build: `docker build -t mp216-api:1.0 .`

Run: `docker run -d --name mp216-api -p 8000:8000 mp216-api:1.0`

Health: `http://localhost:8000/health`

Logs: `docker logs mp216-api`

## Kubernetes Configuration

- Deployment: `mp216-api`
- Service: `mp216-svc`
- Replicas: 2
- Container port: 8000
- Service port: 80, forwarding to 8000
- Readiness and liveness probes: GET `/health`
- Liveness starts later than readiness

The manifests are under `k8s/`. They were authored
and syntax-checked; we have not claimed to deploy
them on a live Kubernetes cluster.

## GitHub Actions

The workflow in `.github/workflows/ci.yml` runs API
and monitoring tests on pushes to main and pull
requests targeting main.

It uses Ubuntu and Python 3.13 to run automated tests.
The Docker container uses Python 3.12.

The CI workflow performs automated testing.
It does not deploy the application to Kubernetes.

The latest verified workflow was successful.

Local MLflow registry tests require model artifacts
and are therefore kept separate from CI.

## Structured Logging

Successful predictions write a JSON log line to stdout
with an event name and predicted number of minutes.

The custom log does not print customer attributes.

## What to Verify Before Submitting This Document

Compare this document with `Dockerfile`, `app/main.py`,
the two `k8s/` files, and `.github/workflows/ci.yml`.

Confirm the current commands.