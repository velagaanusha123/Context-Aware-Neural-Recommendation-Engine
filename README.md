# 🛍️ Context-Aware Fashion Recommendation Engine

## 📌 Project Overview

A deep learning based fashion recommendation system that provides personalized product recommendations using user behaviour, product metadata, and contextual information.

The system uses a **Two-Tower Neural Retrieval Architecture** to generate user and item embeddings, then performs fast similarity search using FAISS.

---

# ✨ Features

## 🎯 Personalized Recommendations

- Two-Tower Neural Recommendation Model
- User and Item embedding generation
- Context-aware recommendations
- Top-K similar product retrieval
- Seasonal/month based filtering

---

## 📊 Analytics

- User interaction analysis
- Product popularity analysis
- Category trends
- Recommendation insights

---

## 🧠 Model Architecture

User Side:

User ID + Age

↓

User Tower

↓

User Embedding


Item Side:

Item ID + Category + Colour + Garment Type

↓

Item Tower

↓

Item Embedding


Similarity Search:

User Embedding

↓

FAISS

↓

Top-K Products


---

# 🛠️ Tech Stack

- Python
- TensorFlow
- TensorFlow Recommenders
- FastAPI
- FAISS
- Redis
- Pandas
- NumPy
- Plotly
- Locust


---

# 📂 Dataset

H&M Personalized Fashion Recommendations Dataset

Contains:

- User interactions
- Product information
- Transaction history


---

# 🚀 Running Project


## 1. Activate environment

```bash
.\.venv\Scripts\activate