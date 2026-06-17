import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import tensorflow as tf

from retrieval_model import (
    UserModel,
    ItemModel,
    FashionRetrievalModel
)

print("[INFO] Loading dataset...")

# =====================================================
# SAFE PROJECT ROOT (FIXED)
# =====================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "Processed data",
    "model_ready_recommendation_dataset.csv"
)

print("[INFO] Dataset Path:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

# =====================================================
# REQUIRED COLUMNS
# =====================================================

columns = [
    "user_idx",
    "item_idx",
    "age",
    "product_group_name",
    "colour_group_name",
    "garment_group_name"
]

df = df[columns]

for col in columns:
    df[col] = df[col].astype(int)

# =====================================================
# LIMIT DATA (optional for quick test)
# =====================================================

#df = df.head(100000)

# =====================================================
# VOCAB SIZES
# =====================================================

max_user_id = df["user_idx"].max()
max_item_id = df["item_idx"].max()

max_product_group = df["product_group_name"].max()
max_colour_group = df["colour_group_name"].max()
max_garment_group = df["garment_group_name"].max()
max_age = df["age"].max()

print("[INFO] Creating dataset...")

# =====================================================
# TF DATASET (FIXED: proper mapping + shuffle)
# =====================================================

dataset = tf.data.Dataset.from_tensor_slices({
    "user_idx": df["user_idx"].values.astype("int32"),
    "item_idx": df["item_idx"].values.astype("int32"),
    "age": df["age"].values.astype("int32"),
    "product_group_name": df["product_group_name"].values.astype("int32"),
    "colour_group_name": df["colour_group_name"].values.astype("int32"),
    "garment_group_name": df["garment_group_name"].values.astype("int32"),
})

dataset = dataset.shuffle(50000).batch(1024).prefetch(tf.data.AUTOTUNE)
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

# IMPORTANT: candidate dataset must be batched
candidate_dataset = tf.data.Dataset.from_tensor_slices({
    "item_idx": df["item_idx"].values,
    "product_group_name": df["product_group_name"].values,
    "colour_group_name": df["colour_group_name"].values,
    "garment_group_name": df["garment_group_name"].values
}).batch(256)

# =====================================================
# RETRIEVAL MODEL
# =====================================================

model = FashionRetrievalModel(
    user_model,
    item_model,
    candidate_dataset
)

model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01)
)

print("[INFO] Training Started...")

model.fit(
    dataset,
    epochs=10
)

# =====================================================
# SAVE MODEL (FIXED PATH SAFETY)
# =====================================================

MODEL_DIR = os.path.join(BASE_DIR, "model", "two_tower")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model.weights.h5")

print("[INFO] Saving model to:", MODEL_PATH)

model.save_weights(MODEL_PATH)

print("[INFO] Training Complete")
print("[INFO] Model Saved Successfully ✔")