"""Central configuration: paths, dataset identity, modeling constants and the
Dracula palette shared by the pipeline, the serving layer and the dashboard.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = BASE_DIR / "models"

PIPELINE_PATH: Path = MODELS_DIR / "pipeline.joblib"
MODEL_CARD_PATH: Path = MODELS_DIR / "model_card.json"
PROCESSED_PATH: Path = PROCESSED_DIR / "train.parquet"

# The upstream "Give Me Some Credit" Kaggle competition is consent-gated, so the
# prepared file is versioned in full here as the reproducible source.
SAMPLE_FILENAME: str = "credit_risk.csv"
SAMPLE_PATH: Path = SAMPLE_DIR / SAMPLE_FILENAME
KAGGLE_DATASET: str = "competitions/GiveMeSomeCredit"
RAW_FILENAME: str = "credit_risk.csv"

TARGET: str = "target"
POSITIVE_LABEL: int = 1
ID_COLS: tuple[str, ...] = ()
# No target leakage: all features describe the applicant at decision time.

NUMERIC_FEATURES: tuple[str, ...] = (
    "RevolvingUtilization", "Age", "DebtRatio", "MonthlyIncome",
    "OpenCreditLines", "RealEstateLoans", "Dependents",
    "PastDue30_59", "PastDue60_89", "PastDue90",
)
ENGINEERED_FEATURES: tuple[str, ...] = ("TotalPastDue", "IncomePerDependent", "MonthlyIncomeMissing")

TEST_SIZE: float = 0.2
SEED: int = 42
CV_FOLDS: int = 4
TUNING_ITERS: int = 14
SCORING: str = "average_precision"

CONTACT_CAPACITIES: tuple[int, ...] = (1_000, 5_000, 10_000)
DEFAULT_THRESHOLD: float = 0.5

DRACULA = {
    "background": "#282a36", "current_line": "#44475a", "foreground": "#f8f8f2",
    "comment": "#6272a4", "cyan": "#8be9fd", "green": "#50fa7b", "orange": "#ffb86c",
    "pink": "#ff79c6", "purple": "#bd93f9", "red": "#ff5555", "yellow": "#f1fa8c",
}
