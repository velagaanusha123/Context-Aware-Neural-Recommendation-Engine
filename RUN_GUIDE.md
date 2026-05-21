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
