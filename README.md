# 🛍️ Context-Aware Neural Recommendation Engine

## 📌 Project Overview
This is a Deep Learning-based recommendation system that suggests personalized items to users based on their past interactions, metadata, and contextual behavior.

The model uses **Neural Collaborative Filtering (NCF)** and context-aware learning to improve recommendation accuracy compared to traditional methods.

---

## ✨ Features

### 🎯 Personalized Recommendations
- Neural Collaborative Filtering (NCF) based recommender system
- Context-aware user-item modeling
- Seasonal/trend-based recommendation support
- Interactive Streamlit dashboard

### 📊 Analytics & Visualization
- User interaction trends
- Category popularity analysis
- Recommendation performance insights

### 🧠 Model System
- Deep learning-based architecture (PyTorch / TensorFlow)
- Scalable design for real-world applications
- Backend API support for integration

---

## 🛠️ Tech Stack
- Python
- PyTorch / TensorFlow
- FastAPI
- Streamlit
- Pandas, NumPy, Scikit-learn
- Matplotlib, Plotly

---

## 📂 Dataset
- Customer interaction data
- Transaction history data

Dataset Link:
https://drive.google.com/drive/folders/1WEB9rii1URE1zc-orDWpUTtL-CDmFCbJ?usp=sharing

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt




///////////////////////////////////////////////////////////////////////
Final Run Steps

Whenever you want to run the project:

1. Open project
cd C:\Context-Aware-Neural-Recommendation-Engine-main\Context-Aware-Neural-Recommendation-Engine-main
2. Activate environment
.\.venv\Scripts\activate
3. Start backend
python -m uvicorn Backend.main:app --reload
4. Open Swagger
http://127.0.0.1:8000/docs
How to test
Get sample users
GET /users/sample?n=10

Example output:

5
15
17
18
19
...
Get recommendations
{
  "user_idx": 5,
  "top_k": 10,
  "month_filter": 6
}
What to say in viva
Project Title

Context-Aware H&M Recommendation Engine

Objective

Recommend fashion products to users using machine learning and contextual information.

Dataset

H&M Personalized Fashion Recommendations dataset.

Technologies
Python
FastAPI
PyTorch
FAISS
Pandas
NumPy
Working
User Request
      ↓
FastAPI API
      ↓
User Embedding
      ↓
Neural Recommendation Model
      ↓
FAISS Similarity Search
      ↓
Seasonal Filtering
      ↓
Top-K Recommended Products
Features
Personalized recommendations
Fast retrieval using FAISS
Seasonal recommendations using month filter
Analytics endpoints
Item catalogue browsing
REST API interface
One thing I would still verify

Test with a few more users:

{
  "user_idx": 15,
  "top_k": 10,
  "month_filter": null
}
{
  "user_idx": 23,
  "top_k": 10,
  "month_filter": 12
}

If those also return 200 OK, then your system is stable.