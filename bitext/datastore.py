from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import List, Tuple, Dict, Any

import joblib
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np


_MODEL_NAME = "all-MiniLM-L6-v2"
_CACHE_DIR = Path(".bitext_cache"); _CACHE_DIR.mkdir(exist_ok=True)
_EMB_FILE = _CACHE_DIR / "embeddings.pkl"
_IDX_FILE = _CACHE_DIR / "nn_index.pkl"
_INSTRUCTION_COL = "instruction"
_RESPONSE_COL = "response"
_CATEGORY_COL = "category"
_INTENT_COL = "intent"
_FLAGS_COL = "flags"


class _Store:

    def __init__(self) -> None:
        self.df = self._load_df()
        self.model = SentenceTransformer(_MODEL_NAME)
        self.embeddings, self.nn = self._load_or_build_index()

    def get_columns(self) -> List[str]:
        return self.df.columns.tolist()

    def get_categories(self) -> List[str]:
        return sorted(self.df[_CATEGORY_COL].unique().tolist())

    def get_intents(self) -> List[str]:
        return sorted(self.df[_INTENT_COL].unique().tolist())

    def get_flags(self) -> List[str]:
        return sorted(self.df[_FLAGS_COL].unique().tolist())

    @staticmethod
    def _load_df() -> pd.DataFrame:
        ds = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"
        )
        df = pd.DataFrame(ds)
        
        # Validate required columns exist
        required_cols = [_CATEGORY_COL, _INTENT_COL, _INSTRUCTION_COL, _RESPONSE_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")
            
        return df

    def _load_or_build_index(self):
        if _EMB_FILE.exists() and _IDX_FILE.exists():
            emb = joblib.load(_EMB_FILE)
            nn = joblib.load(_IDX_FILE)
            return emb, nn

        texts = self.df[_INSTRUCTION_COL].astype(str) + " " + self.df[_RESPONSE_COL].astype(str)
        emb = self.model.encode(texts.tolist(), show_progress_bar=True, batch_size=64)
        nn = NearestNeighbors(metric="cosine", n_neighbors=5).fit(emb)

        joblib.dump(emb, _EMB_FILE)
        joblib.dump(nn, _IDX_FILE)
        return emb, nn


_store = _Store()