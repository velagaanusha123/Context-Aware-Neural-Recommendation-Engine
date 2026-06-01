import os
import pandas as pd

from redis_client import redis_client
print("Loading users...")

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "Processed data",
    "model_ready_recommendation_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

user_df = df[
    ["user_idx", "age"]
].drop_duplicates()

count = 0

for _, row in user_df.iterrows():

    redis_client.hset(
        f"user:{int(row['user_idx'])}",
        mapping={
            "age": int(row["age"])
        }
    )

    count += 1

print(f"[INFO] Loaded {count} users into Redis")