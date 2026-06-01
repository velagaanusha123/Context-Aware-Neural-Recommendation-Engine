import os

print("🚀 ML PIPELINE STARTED")

def refresh_data():
    print("1️⃣ Refreshing dataset...")


def train_model():
    print("2️⃣ Training Two-Tower Model...")
    os.system("python Backend/two_tower/train.py")


def generate_embeddings():
    print("3️⃣ Generating embeddings...")


def build_faiss_index():
    print("4️⃣ Building FAISS index...")
    os.system("python Backend/recommender/build_faiss.py")


if __name__ == "__main__":

    refresh_data()
    train_model()
    generate_embeddings()
    build_faiss_index()

    print("✅ PIPELINE COMPLETED SUCCESSFULLY")