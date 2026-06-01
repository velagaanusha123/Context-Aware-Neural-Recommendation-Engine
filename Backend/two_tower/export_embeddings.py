import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf

from retrieval_model import (
    UserModel,
    ItemModel,
    FashionRetrievalModel
)

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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "two_tower",
    "model.weights.h5"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "embeddings"
)

os.makedirs(EMBEDDING_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("[INFO] Loading dataset...")

df = pd.read_csv(DATA_PATH)

for col in [
    "user_idx",
    "item_idx",
    "age",
    "product_group_name",
    "colour_group_name",
    "garment_group_name"
]:
    df[col] = df[col].astype(int)

# =====================================================
# MODEL PARAMETERS
# =====================================================

max_user_id = int(df["user_idx"].max())
max_item_id = int(df["item_idx"].max())

max_age = int(df["age"].max())

max_product_group = int(df["product_group_name"].max())
max_colour_group = int(df["colour_group_name"].max())
max_garment_group = int(df["garment_group_name"].max())

# =====================================================
# BUILD MODELS
# =====================================================

user_model = UserModel(
    max_user_id=max_user_id,
    max_age=max_age
)

item_model = ItemModel(
    max_item_id=max_item_id,
    max_product_group=max_product_group,
    max_colour_group=max_colour_group,
    max_garment_group=max_garment_group
)

candidate_dataset = tf.data.Dataset.from_tensor_slices({
    "item_idx": df["item_idx"].values,
    "product_group_name": df["product_group_name"].values,
    "colour_group_name": df["colour_group_name"].values,
    "garment_group_name": df["garment_group_name"].values
})

model = FashionRetrievalModel(
    user_model,
    item_model,
    candidate_dataset
)

# Build towers manually

user_model({
    "user_idx": tf.constant([0]),
    "age": tf.constant([0])
})

item_model({
    "item_idx": tf.constant([0]),
    "product_group_name": tf.constant([0]),
    "colour_group_name": tf.constant([0]),
    "garment_group_name": tf.constant([0])
})

# =====================================================
# LOAD TRAINED WEIGHTS
# =====================================================

print("[INFO] Loading trained weights...")

model.load_weights(MODEL_PATH)

print("[INFO] Weights Loaded Successfully")

# =====================================================
# ITEM EMBEDDINGS
# =====================================================

item_df = df[
    [
        "item_idx",
        "product_group_name",
        "colour_group_name",
        "garment_group_name"
    ]
].drop_duplicates()

inputs = {
    "item_idx":
        tf.constant(item_df["item_idx"].values),

    "product_group_name":
        tf.constant(item_df["product_group_name"].values),

    "colour_group_name":
        tf.constant(item_df["colour_group_name"].values),

    "garment_group_name":
        tf.constant(item_df["garment_group_name"].values)
}

print("[INFO] Generating embeddings...")

embeddings = item_model(inputs).numpy()

# =====================================================
# SAVE
# =====================================================

np.save(
    os.path.join(
        EMBEDDING_DIR,
        "item_embeddings.npy"
    ),
    embeddings
)

np.save(
    os.path.join(
        EMBEDDING_DIR,
        "item_ids.npy"
    ),
    item_df["item_idx"].values
)

print("[INFO] Embeddings Saved")
print("[INFO] Shape:", embeddings.shape)