"""Acquisition layer. The upstream Kaggle competition is consent-gated, so the
prepared dataset is versioned under data/sample and used directly.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, raw_dir: Path = config.RAW_DIR, sample_path: Path = config.SAMPLE_PATH) -> None:
        self.raw_dir = raw_dir
        self.sample_path = sample_path
        self.raw_path = raw_dir / config.RAW_FILENAME

    def download(self, force: bool = False) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.raw_path.exists() and not force:
            return self.raw_path
        if not self.sample_path.exists():
            raise FileNotFoundError(
                f"Prepared dataset missing at {self.sample_path}; the upstream Kaggle "
                "competition GiveMeSomeCredit requires manual consent and download.")
        shutil.copyfile(self.sample_path, self.raw_path)
        logger.info("Prepared dataset materialized at %s", self.raw_path)
        return self.raw_path

    def load(self) -> pd.DataFrame:
        path = self.raw_path if self.raw_path.exists() else self.download()
        df = pd.read_csv(path)
        logger.info("Loaded %d rows x %d cols from %s", df.shape[0], df.shape[1], path)
        return df
