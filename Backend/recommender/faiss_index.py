import os
import faiss
import numpy as np

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

INDEX_PATH = os.path.join(
    BASE_DIR,
    "Backend",
    "recommender",
    "fashion.index"
)

ITEM_IDS_PATH = os.path.join(
    BASE_DIR,
    "embeddings",
    "item_ids.npy"
)

# =====================================================
# LOAD INDEX
# =====================================================

index = faiss.read_index(INDEX_PATH)

item_ids = np.load(ITEM_IDS_PATH)

print("[INFO] FAISS Index Loaded")
print("[INFO] Total Items:", len(item_ids))


# =====================================================
# SEARCH
# =====================================================

def search(query_embedding, k=10):

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        k
    )

    recommendations = item_ids[
        indices[0]
    ]

    return recommendations


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    random_vector = np.random.rand(
        1,
        index.d
    ).astype(np.float32)

    results = search(
        random_vector,
        k=10
    )

    print("\nTop Recommendations:")
    print(results) 