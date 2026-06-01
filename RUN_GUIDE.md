## 🌟 Project Summary
This project is an **intelligent fashion recommender system** built using the H&M Personalized Fashion Recommendations dataset.
- **Neural Learning**: Uses a Neural Collaborative Filtering (NCF) model to predict customer preferences.
- **Context-Aware**: Allows filtering recommendations by **Month** (Seasonal context) to provide relevant fashion picks for specific times of the year.

## 🚀 Quick Start Steps

### 1. Open Terminal in VS Code
Open the project folder in VS Code. Go to **Terminal -> New Terminal**.

### 2. Start the Backend API
If you have trouble activating the virtual environment, you can run the server directly using this command:
```powershell
.\venv\Scripts\python.exe Backend/main.py
```
*Wait until you see the message: `INFO: Uvicorn running on http://0.0.0.0:8000`.*

### 3. Launch the Frontend
1. Open your browser (Chrome/Edge).
2. Press **Ctrl + O**.
3. Select the file: `Frontend/index.html` from the project folder.
4. Look for the **Green Dot** (API Online) in the top right corner.

---


## 🛠️ Troubleshooting

- **"Term not recognized" for Activate?** 
  If `.\venv\Scripts\Activate.ps1` fails, simply use the direct command in Step 2 above.
- **Port already in use?**
  If you see an error about the socket address, it means the server is already running. Close any other terminal windows running the backend or restart VS Code.
- **Weights only load failed?**
  I have updated the code in `Backend/recommender.py` to fix this. It now uses `weights_only=False` to ensure the pre-trained model loads correctly on all systems.

--------------------------------------------------------------------------------------------------------------------------------------------------

Perfect — I’ll give you two things in a clean, real-world way:

---

# 🚀 PART 1 — HOW TO RUN YOUR FULL PROJECT (STEP-BY-STEP)

You will always run your project in this exact order.

---

# ✅ STEP 0 — Open Project

```bash id="run0"
cd C:\Context-Aware-Neural-Recommendation-Engine-main\Context-Aware-Neural-Recommendation-Engine-main
```

---

# ✅ STEP 1 — Activate Virtual Environment

```bash id="run1"
.\.venv\Scripts\activate
```

You should see:

```text id="run1o"
(.venv)
```

---

# ✅ STEP 2 — Start Redis Server (IMPORTANT)

If Redis is not already running:

```bash id="run2"
redis-server
```

OR check:

```bash id="run2b"
redis-cli ping
```

Expected:

```text id="run2o"
PONG
```

---

# ✅ STEP 3 — Load Users into Redis

```bash id="run3"
python Backend/redis_store/load_user_profiles.py
```

Expected:

```text id="run3o"
[INFO] Redis Connected Successfully
[INFO] Loaded 86915 users into Redis
```

---

# ✅ STEP 4 — Run FAISS Index

```bash id="run4"
python Backend/recommender/faiss_index.py
```

Expected:

```text id="run4o"
[INFO] FAISS Index Loaded
[INFO] Total Items: 26113
```

---

# ✅ STEP 5 — Start FastAPI Server (MAIN SYSTEM)

```bash id="run5"
python -m uvicorn Backend.api.main:app --reload
```

Open:

```text id="run5o"
http://127.0.0.1:8000/docs
```

Test:

```
/recommend?user_id=222851&k=10
```

---

# ✅ STEP 6 — Run Load Test (Optional but IMP)

Open new terminal:

```bash id="run6"
.\.venv\Scripts\activate
python -m locust -f tests/load_test.py
```

Open:

```text id="run6o"
http://localhost:8089
```

---

# ✅ STEP 7 — Run Pipeline (SIMULATION)

```bash id="run7"
python Backend/airflow/retrain_pipeline.py
```

---

# 🎯 FINAL SYSTEM FLOW (what YOU explain)

```text id="flow"
User Request
   ↓
FastAPI (/recommend)
   ↓
Redis (User Profile)
   ↓
User Embedding (Two-Tower Model)
   ↓
FAISS Similarity Search
   ↓
Top-K Products Returned
```