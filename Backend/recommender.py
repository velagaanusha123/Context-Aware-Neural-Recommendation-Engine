"""
recommender.py
--------------
Loads the real H&M processed dataset and builds a lightweight collaborative-
filtering recommender using cosine similarity on the user-item interaction matrix.
"""

import os
import sys

# Add the project root to sys.path so 'Backend' can be imported when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from Backend.neural_model import NCFModel
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from typing import List, Dict, Any, Optional

# --------------------------------------------------------------------------- #
# Paths (relative to the project root where uvicorn is launched from)
# --------------------------------------------------------------------------- #
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_READY = os.path.join(_BASE, "Data", "Processed data",
                            "model_ready_recommendation_dataset.csv")
_ARTICLES    = os.path.join(_BASE, "Data", "Raw data", "articles_hm.csv")
_MODEL_PATH  = os.path.join(_BASE, "model", "ncf_model.pth")


# --------------------------------------------------------------------------- #
# Data loading (done once at import time)
# --------------------------------------------------------------------------- #

def _load_data():
    """Load and prepare the interaction matrix + article lookup table."""
    print("[Recommender] Loading interaction dataset ...")
    df = pd.read_csv(_MODEL_READY)

    # Keep only the columns we need for the model
    interactions = df[["user_idx", "item_idx", "interaction",
                        "price", "month", "age",
                        "product_type_name", "product_group_name",
                        "colour_group_name", "section_name",
                        "garment_group_name"]].copy()

    # Build a de-duplicated user-item matrix (sum interactions per pair)
    agg = (interactions.groupby(["user_idx", "item_idx"])["interaction"]
                       .sum().reset_index())

    n_users = agg["user_idx"].max() + 1
    n_items = agg["item_idx"].max() + 1

    sparse_matrix = csr_matrix(
        (agg["interaction"].values,
         (agg["user_idx"].values, agg["item_idx"].values)),
        shape=(n_users, n_items)
    )

    print(f"[Recommender] Matrix shape: {sparse_matrix.shape}")

    # ------------------------------------------------------------------ #
    # Article name lookup  (item_idx is the positional index in articles)
    # ------------------------------------------------------------------ #
    print("[Recommender] Loading articles catalogue ...")
    articles = pd.read_csv(_ARTICLES,
                           usecols=["article_id", "prod_name",
                                    "product_type_name", "product_group_name",
                                    "colour_group_name", "section_name",
                                    "garment_group_name"])

    # item_idx in the model_ready dataset is the row position (0-based)
    # so we can use iloc directly
    articles = articles.reset_index(drop=True)   # ensures 0-based positional index

    # Build a dict  item_idx -> article metadata
    item_meta: Dict[int, Dict[str, Any]] = {}
    for idx, row in articles.iterrows():
        item_meta[int(idx)] = {
            "item_idx":           int(idx),
            "article_id":         int(row["article_id"]),
            "prod_name":          str(row["prod_name"]),
            "product_type":       str(row["product_type_name"]),
            "product_group":      str(row["product_group_name"]),
            "colour":             str(row["colour_group_name"]),
            "section":            str(row["section_name"]),
            "garment_group":      str(row["garment_group_name"]),
        }


    # Popularity scores (purchase count per item across all users)
    popularity = (agg.groupby("item_idx")["interaction"]
                     .sum()
                     .reset_index()
                     .rename(columns={"interaction": "pop_score"}))
    pop_dict = dict(zip(popularity["item_idx"], popularity["pop_score"]))

    # Sample of valid user IDs (for the UI demo)
    sample_users = sorted(agg["user_idx"].unique()[:500].tolist())

    # Extra per-interaction metadata for context filtering
    context = interactions.copy()

    return sparse_matrix, item_meta, pop_dict, sample_users, context


# Load once
_MATRIX, _ITEM_META, _POP, _SAMPLE_USERS, _CONTEXT = _load_data()

# --------------------------------------------------------------------------- #
# Neural Model Loading
# --------------------------------------------------------------------------- #

_NEURAL_MODEL: Optional[NCFModel] = None

def _load_neural_model():
    global _NEURAL_MODEL
    if os.path.exists(_MODEL_PATH):
        try:
            print(f"[Recommender] Loading Neural Model from {_MODEL_PATH} ...")
            checkpoint = torch.load(_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
            model = NCFModel(checkpoint['n_users'], checkpoint['n_items'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            _NEURAL_MODEL = model
            print("[Recommender] Neural Model Loaded Done")
        except Exception as e:
            print(f"[Recommender] Error loading neural model: {e}")
    else:
        print("[Recommender] Neural model not found. Using Similarity fallback.")

_load_neural_model()
print("[Recommender] Ready Done")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def get_sample_users(n: int = 100) -> List[int]:
    """Return up to *n* sample user IDs that exist in the dataset."""
    return _SAMPLE_USERS[:n]


def get_item_catalogue(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Return a slice of the article catalogue."""
    keys = list(_ITEM_META.keys())[offset: offset + limit]
    items = []
    for k in keys:
        meta = dict(_ITEM_META[k])
        meta["popularity"] = int(_POP.get(k, 0))
        items.append(meta)
    return items


def recommend_for_user(
    user_idx: int,
    top_k: int = 10,
    month_filter: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Return top-K recommendations for *user_idx*.

    Strategy:
    1. If Neural Model exists: Use it to score items.
    2. Else: Use Similarity-based scoring.
    3. Zero out already-seen items and apply filters.
    """
    n_users = _MATRIX.shape[0]
    n_items = _MATRIX.shape[1]

    if user_idx < 0 or user_idx >= n_users:
        raise ValueError(f"user_idx {user_idx} out of range [0, {n_users})")

    user_vec = _MATRIX[user_idx]
    seen_items = set(user_vec.indices)

    # ─── NEURAL SCORING ───
    if _NEURAL_MODEL is not None:
        # For simplicity in demo, we score the most popular 500 items + some randoms
        # Scoring all 50k items every time is slow for a live demo
        candidate_items = np.array(list(_POP.keys()))
        
        # Sort candidates by popularity first to get top items
        candidate_items = sorted(candidate_items, key=lambda x: _POP.get(x, 0), reverse=True)[:1000]
        candidate_items = torch.LongTensor(candidate_items)
        user_tensor = torch.LongTensor([user_idx] * len(candidate_items))
        
        with torch.no_grad():
            scores_tensor = _NEURAL_MODEL(user_tensor, candidate_items)
            scores = scores_tensor.numpy()
            item_indices = candidate_items.numpy()
            
        # Create a full score vector
        full_scores = np.zeros(n_items)
        for i, idx in enumerate(item_indices):
            full_scores[idx] = scores[i]
            
    # ─── SIMILARITY FALLBACK ───
    else:
        sim_scores = cosine_similarity(user_vec, _MATRIX).flatten()
        sim_scores[user_idx] = 0.0
        top_neighbours = np.argsort(sim_scores)[::-1][:50]
        weights = sim_scores[top_neighbours]
        neighbour_matrix = _MATRIX[top_neighbours]
        full_scores = np.array(neighbour_matrix.T.dot(weights)).flatten()

    # Zero out already-seen items
    for item_idx in seen_items:
        if item_idx < len(full_scores):
            full_scores[item_idx] = 0.0

    # Optional month filter
    if month_filter is not None:
        allowed = set(_CONTEXT.loc[_CONTEXT["month"] == month_filter, "item_idx"].unique())
        mask = np.zeros(len(full_scores), dtype=bool)
        for i in allowed:
            if i < len(mask): mask[i] = True
        full_scores = full_scores * mask

    # Top-K
    top_items = np.argsort(full_scores)[::-1][:top_k]

    results = []
    for item_idx in top_items:
        if full_scores[item_idx] <= 0: break
        meta = dict(_ITEM_META.get(int(item_idx), {"item_idx": int(item_idx), "prod_name": "Unknown"}))
        meta["score"] = float(round(full_scores[item_idx], 4))
        meta["popularity"] = int(_POP.get(int(item_idx), 0))
        results.append(meta)

    return results
