import os
import faiss
import numpy as np

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

EMBEDDING_PATH = os.path.join(
    BASE_DIR,
    "embeddings",
    "item_embeddings.npy"
)

INDEX_PATH = os.path.join(
    BASE_DIR,
    "Backend",
    "recommender",
    "fashion.index"
)

print("[INFO] Loading embeddings...")

embeddings = np.load(EMBEDDING_PATH).astype("float32")

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

print("[INFO] FAISS Index Created")
print("[INFO] Total vectors:", index.ntotal)