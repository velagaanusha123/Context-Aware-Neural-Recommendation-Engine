import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "Processed data",
    "model_ready_recommendation_dataset.csv"
)

EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "embeddings",
    "item_embeddings.npy"
)

ITEM_IDS_PATH = os.path.join(
    BASE_DIR,
    "embeddings",
    "item_ids.npy"
)

# =====================================================
# LOAD DATA
# =====================================================

print("[INFO] Loading data...")

df = pd.read_csv(DATA_PATH)

item_embeddings = np.load(EMBEDDINGS_PATH)
item_ids = np.load(ITEM_IDS_PATH)

# Normalize
item_embeddings = (
    item_embeddings /
    np.linalg.norm(
        item_embeddings,
        axis=1,
        keepdims=True
    )
)

# =====================================================
# USER HISTORY
# =====================================================

user_groups = df.groupby(
    "user_idx"
)["item_idx"].apply(list)

users = list(user_groups.index)[:500]

item_id_to_index = {
    item_id: idx
    for idx, item_id in enumerate(item_ids)
}

# =====================================================
# METRICS
# =====================================================

def recall_at_k(actual, predicted, k):

    predicted = predicted[:k]

    hits = len(
        set(actual) &
        set(predicted)
    )

    return hits / max(len(actual), 1)


def ndcg_at_k(actual, predicted, k):

    predicted = predicted[:k]

    dcg = 0

    for i, item in enumerate(predicted):

        if item in actual:
            dcg += (
                1 /
                np.log2(i + 2)
            )

    idcg = sum(
        1 / np.log2(i + 2)
        for i in range(
            min(len(actual), k)
        )
    )

    if idcg == 0:
        return 0

    return dcg / idcg

# =====================================================
# EVALUATION
# =====================================================

print("[INFO] Running evaluation...")

recall10 = []
recall20 = []

ndcg10 = []
ndcg20 = []

for user in users:

    history = user_groups[user]

    if len(history) < 2:
        continue

    test_item = history[-1]

    if test_item not in item_id_to_index:
        continue

    train_items = history[:-1]

    train_indices = [
        item_id_to_index[i]
        for i in train_items
        if i in item_id_to_index
    ]

    if len(train_indices) == 0:
        continue

    # User profile
    user_embedding = np.mean(
        item_embeddings[train_indices],
        axis=0
    )

    user_embedding = (
        user_embedding /
        np.linalg.norm(user_embedding)
    )

    # Similarity
    scores = np.dot(
        item_embeddings,
        user_embedding
    )

    top_idx = np.argsort(
        -scores
    )[:100]

    predicted_items = item_ids[top_idx]

    recall10.append(
        recall_at_k(
            [test_item],
            predicted_items,
            10
        )
    )

    recall20.append(
        recall_at_k(
            [test_item],
            predicted_items,
            20
        )
    )

    ndcg10.append(
        ndcg_at_k(
            [test_item],
            predicted_items,
            10
        )
    )

    ndcg20.append(
        ndcg_at_k(
            [test_item],
            predicted_items,
            20
        )
    )

# =====================================================
# RESULTS
# =====================================================

print("\n========== RESULTS ==========")

print(
    f"Recall@10 : {np.mean(recall10):.4f}"
)

print(
    f"Recall@20 : {np.mean(recall20):.4f}"
)

print(
    f"NDCG@10   : {np.mean(ndcg10):.4f}"
)

print(
    f"NDCG@20   : {np.mean(ndcg20):.4f}"
)