.PHONY: install setup profile train analyze test api mlflow-ui

# One command to go from a fresh clone to a trained model.
# Creates a virtualenv, installs every dependency group, profiles the data
# and trains the model — everything a recruiter needs before opening the API.
setup:
	python -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt -r requirements-mlflow.txt -r requirements-api.txt
	.venv/bin/python -m pip install -e .
	PYTHONPATH=src .venv/bin/python -m mlflow_credit_risk.cli profile
	PYTHONPATH=src .venv/bin/python -m mlflow_credit_risk.cli train
	@echo ""
	@echo "Setup complete. Activate the environment with 'source .venv/bin/activate', then run 'make api' or 'make mlflow-ui'."

install:
	python -m pip install -r requirements.txt

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

profile:
	PYTHONPATH=src python -m mlflow_credit_risk.cli profile

train:
	PYTHONPATH=src python -m mlflow_credit_risk.cli train

analyze:
	PYTHONPATH=src python -m mlflow_credit_risk.cli analyze

test:
	python -m pytest

api:
	PYTHONPATH=src uvicorn mlflow_credit_risk.api:app --reload
