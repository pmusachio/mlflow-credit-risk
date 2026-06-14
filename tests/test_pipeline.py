"""Smoke tests for the data contract, engineering and the serving surface."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402
from src.preprocessing import FeaturePrep, Preprocessor, engineer  # noqa: E402

APPLICANT = {"RevolvingUtilization": 0.5, "Age": 45, "DebtRatio": 0.3, "MonthlyIncome": 5000.0,
             "OpenCreditLines": 8, "RealEstateLoans": 1, "Dependents": 2,
             "PastDue30_59": 0, "PastDue60_89": 0, "PastDue90": 0}


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(config.SAMPLE_PATH)


def test_target_present_and_imbalanced(sample):
    assert config.TARGET in sample.columns
    assert 0.02 < sample[config.TARGET].mean() < 0.20


def test_engineering_and_contract(sample):
    X, y = Preprocessor().run(sample)
    assert config.TARGET not in X.columns
    eng = engineer(sample.head(100))
    for c in config.ENGINEERED_FEATURES:
        assert c in eng.columns
    assert set(y.unique()) <= {0, 1}


def test_feature_prep_fixed_columns(sample):
    a = FeaturePrep().fit_transform(sample.head(20))
    b = FeaturePrep().transform(sample.head(5))
    assert list(a.columns) == list(b.columns)


def test_predictor_contract():
    pred = Predictor()
    s = pred.score_one(APPLICANT)
    assert 0.0 <= s <= 1.0
    assert pred.decision(0.9).startswith("decline")
    assert pred.decision(0.01) == "approve"
    assert len(pred.top_features(5)) >= 1


def test_delinquent_scores_above_clean():
    pred = Predictor()
    risky = {**APPLICANT, "PastDue90": 3, "PastDue30_59": 4, "RevolvingUtilization": 0.98, "MonthlyIncome": 1500.0}
    assert pred.score_one(risky) > pred.score_one(APPLICANT)
