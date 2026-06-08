# Delivery and consumption

The project is structured to version experiments, promote a model to Production, and serve predictions through an API. SageMaker is documented as an enterprise option, but it is not the most cost-effective path for a portfolio demo.

## Channels

- **MLflow Tracking:** compare runs, parameters, and metrics.
- **MLflow Model Registry:** promote the best model to `Production`.
- **FastAPI:** `/predict` endpoint for local or cloud consumption.
- **SageMaker:** `scripts/sagemaker_deployment_template.py` with a base configuration to deploy the registered model.
- **Lower-cost alternatives:** DagsHub for remote tracking, and Render/Railway/Fly.io/Cloud Run/Hugging Face Spaces to publish the API or a demo.

## Viewing the MLflow UI

Use two terminals from the project root.

Terminal 1: start the UI.

```bash
python -m pip install -r requirements.txt -r requirements-mlflow.txt
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open in the browser:

```text
http://127.0.0.1:5000
```

Terminal 2: generate a tracked run.

```bash
PYTHONPATH=src python -m mlflow_credit_risk.cli train
```

In the UI, select the `08_mlflow_credit_risk` experiment, open the most recent run, and check:

- **Parameters:** project configuration, target, and problem family.
- **Metrics:** ROC AUC, average precision, F1, and other logged metrics.
- **Artifacts:** the `models/model.joblib` file and the sklearn model artifacts.

If the run does not appear, confirm that the training command was executed from the repository root and that the UI is pointing at `./mlruns`.

## Local API

```bash
python -m pip install -r requirements.txt -r requirements-api.txt
PYTHONPATH=src uvicorn mlflow_credit_risk.api:app --reload
```

## SageMaker

```bash
python scripts/sagemaker_deployment_template.py --model-uri models:/mlflow-credit-risk/Production
```

The script runs in dry-run mode by default. Use `--execute` only after configuring AWS credentials, a SageMaker role, and a tracking URI.

## Cheaper alternatives than AWS

For a portfolio, the practical recommendation is to separate tracking from inference:

- **Local tracking:** zero cost; run `mlflow ui` locally and capture screenshots for the README.
- **DagsHub + MLflow:** good for remote tracking without maintaining your own server. Set `MLFLOW_TRACKING_URI`, username, and DagsHub token, then run the same training command.
- **Render:** simple way to publish the FastAPI service on a free/low-cost plan, good for demos that can sleep when idle.
- **Railway:** good for quick API deployment with environment variables and simple logs.
- **Fly.io:** good when you want Docker and a small machine with predictable cost.
- **Google Cloud Run:** good for an API with scale-to-zero and pay-per-use billing.
- **Hugging Face Spaces:** good for a public demo, especially if you turn the API into a Docker or Streamlit app.

Cost recommendation for this project: keep the MLflow UI local or on DagsHub and publish only the API on Render, Cloud Run, or Fly.io. Use SageMaker only if the goal is to demonstrate a managed deployment on the AWS stack.

Example of remote tracking with DagsHub:

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repository>.mlflow"
export MLFLOW_TRACKING_USERNAME="<user>"
export MLFLOW_TRACKING_PASSWORD="<token>"
PYTHONPATH=src python -m mlflow_credit_risk.cli train
```

Useful links to check current plans:

- [MLflow Tracking UI](https://mlflow.org/docs/latest/ml/tracking/)
- [DagsHub MLflow](https://dagshub.com/docs/integration_guide/mlflow_tracking/)
- [Render Pricing](https://render.com/pricing)
- [Railway Pricing](https://railway.com/pricing)
- [Fly.io Pricing](https://fly.io/docs/about/pricing/)
- [Google Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Hugging Face Spaces Pricing](https://huggingface.co/pricing)
