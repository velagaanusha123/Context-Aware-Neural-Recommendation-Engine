
import faiss
import numpy as np

def build_index(item_embeddings):
    dim = item_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(item_embeddings).astype("float32"))
    return index

def search(index, user_vector, k=10):
    D, I = index.search(np.array([user_vector]).astype("float32"), k)
    return I[0]
