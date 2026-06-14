"""Transformation layer. Feature engineering (total past-due count, income per
dependent, income-missing flag) lives in a custom first pipeline step so training
and serving share the identical transform.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config

logger = logging.getLogger(__name__)

SOURCE_COLUMNS = list(config.NUMERIC_FEATURES)
MODEL_FEATURES = list(config.NUMERIC_FEATURES) + ["TotalPastDue", "IncomePerDependent", "MonthlyIncomeMissing"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in config.NUMERIC_FEATURES:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["MonthlyIncomeMissing"] = out["MonthlyIncome"].isna().astype(float)
    out["TotalPastDue"] = (out.get("PastDue30_59", 0).fillna(0)
                           + out.get("PastDue60_89", 0).fillna(0)
                           + out.get("PastDue90", 0).fillna(0))
    out["IncomePerDependent"] = out["MonthlyIncome"].fillna(0) / (out["Dependents"].fillna(0) + 1)
    return out


class FeaturePrep(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        df = engineer(pd.DataFrame(X).copy())
        for c in MODEL_FEATURES:
            if c not in df.columns:
                df[c] = np.nan
        return df[MODEL_FEATURES]


def build_column_transformer() -> ColumnTransformer:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([("num", numeric_pipe, MODEL_FEATURES)], remainder="drop")


class Preprocessor:
    def __init__(self, processed_path=config.PROCESSED_PATH) -> None:
        self.processed_path = processed_path

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if config.TARGET not in df.columns:
            raise ValueError(f"Target '{config.TARGET}' missing")
        y = df[config.TARGET].astype(int)
        X = df[[c for c in SOURCE_COLUMNS if c in df.columns]].copy()
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = engineer(df).copy()
        cleaned[config.TARGET] = y.values
        cleaned.to_parquet(self.processed_path, index=False)
        logger.info("Processed frame (%d rows) written to %s", len(cleaned), self.processed_path)
        return X, y
