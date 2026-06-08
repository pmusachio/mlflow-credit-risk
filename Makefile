.PHONY: install profile train analyze test api

install:
	python -m pip install -r requirements.txt

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
