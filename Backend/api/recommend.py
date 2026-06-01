from Backend.redis_store.redis_client import redis_client
from Backend.recommender.faiss_index import search
from Backend.two_tower.retrieval_model import UserModel
import tensorflow as tf
import json

# Load user ID to index mapping (optional)
try:
    with open("user_mapping.json") as f:
        user_to_idx = json.load(f)
except FileNotFoundError:
    user_to_idx = {}

def recommend(user_id, k=10):
    """Return top-k product recommendations for a given user.

    Args:
        user_id (int or str): External user identifier supplied by the client.
        k (int): Number of recommendations to return.
    """
    # Retrieve user profile from Redis
    user_data = redis_client.hgetall(f"user:{user_id}")
    if not user_data:
        return {"error": "User not found"}

    age = int(user_data["age"])

    # Translate external user_id to internal model index if mapping exists
    user_idx = user_to_idx.get(str(user_id))
    if user_idx is None:
        # Fallback: assume the supplied ID is already the model index
        try:
            user_idx = int(user_id)
        except ValueError:
            return {"error": "Unknown user_id in model mapping"}

    # Adjust max_user_id to a sufficiently large value (e.g., 300000) to cover dataset indices
    user_model = UserModel(max_user_id=300000, max_age=100)

    # Compute user embedding using the two‑tower model
    user_embedding = user_model({
        "user_idx": tf.constant([user_idx]),
        "age": tf.constant([age])
    }).numpy()

    # Perform similarity search via FAISS
    results = search(user_embedding, k=k)
    return results.tolist()