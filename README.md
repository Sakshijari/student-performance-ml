## Student Performance Prediction – Project Overview

This project uses supervised machine learning to **predict student academic performance** (pass/fail) from a public student performance dataset.  
The goal is to identify **at‑risk students early** so that universities can provide timely support and improve retention.

### Project Structure

```text
student-performance-ml/
├─ data/
│  └─ StudentsPerformance.csv        # Input dataset
├─ src/
│  ├─ __init__.py
│  ├─ data_preprocessing.py          # Loading, target creation, encoding, scaling, splits
│  ├─ train_and_evaluate.py          # Train & evaluate all models, save metrics and models
│  ├─ run_pipeline.py                # Main entry point for the full pipeline
├─ graph/
│  └─ *.png                          # Architecture figures
├─ output/
│  ├─ evaluation_results.json        # Metrics for all models
│  └─ models/                        # Saved trained models (.joblib)
├─ venv/                             # Python virtual environment
├─ requirements.txt                  # Python dependencies
├─ README.md                         # This file
└─ .gitignore
```

### Models and Methods

- **Models used:**
  - Logistic Regression (baseline, interpretable)
  - Random Forest (non‑linear patterns, feature importance)
  - Support Vector Machine – SVM with RBF kernel
  - Gradient Boosting Classifier
- **Target:** binary label (pass/fail) based on the average of math, reading and writing scores.  
- **Preprocessing:** one‑hot encoding for categorical features, standard scaling for numerical features, stratified train/validation/test splits.  
- **Evaluation:** Accuracy, Precision, Recall, F1‑score, ROC‑AUC, confusion matrix, and 5‑fold stratified cross‑validation on F1.

## Environment Setup (first time)

Create a virtual environment and install dependencies:

```powershell
# From the project root of this project

# 1) Create virtual environment (Windows)
python -m venv venv

# 2) Activate it
.\venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
```

After this one‑time setup, you can follow the run instructions below.

## Student Performance Prediction – How to Run

### Step 1: Open a terminal in the project folder

In VS Code/Cursor: **Terminal → New Terminal**, or open PowerShell/Command Prompt in the project folder:

### Step 2: Run the full pipeline

**Option A – Use venv Python directly (easiest on Windows):**

```powershell
.\venv\Scripts\python.exe -m src.run_pipeline
```

**Option B – Activate venv first, then run:**

```powershell
.\venv\Scripts\activate
python -m src.run_pipeline
```

### Step 3: What you should see

- **Console:** “Loading data…” → “Preprocessing…” → “Training and evaluating…” → a **MODEL COMPARISON** table with Accuracy, Precision, Recall, F1, ROC-AUC and Confusion Matrix for each of the 4 models, then **5-Fold Stratified CV (F1)** summary, and finally “Results and models saved to: …\\output”.
- **No Python errors** (exit code 0).

### Step 4: Verify output files

After a successful run, check that these exist:

| What | Where |
|------|--------|
| Evaluation metrics (JSON) | `output/evaluation_results.json` |
| Trained models | `output/models/Logistic_Regression.joblib`, `Random_Forest.joblib`, `SVM.joblib`, `Gradient_Boosting.joblib` |

If all four models appear in the console table and these files are present, **everything is running correctly**.

### Quick one-liner (from project folder)

```powershell
.\venv\Scripts\python.exe -m src.run_pipeline
```
---