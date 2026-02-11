# Phase 2 – Preparation Summary

This folder contains everything prepared for **Phase 2 (Development & Reflection)** of the Data Science Use Case portfolio, addressing the professor’s Phase 1 feedback.

---

## What Was Added (Feedback Addressed)

1. **System architecture and workflow**  
   - Documented in `docs/Phase2_Document_Jariwala_Sakshi_4243407.md` (Section 5).  
   - Mermaid diagrams in `docs/phase2_workflow_diagram.md` (export to PNG/SVG via [mermaid.live](https://mermaid.live)).

2. **Timeline and task breakdown for Phase 2**  
   - Detailed 6-week plan in the Phase 2 document (Section 4): environment, preprocessing, baseline + RF, SVM + GB, evaluation, documentation.

3. **Data privacy and ethical handling of student data**  
   - Section 7 in the Phase 2 document: purpose limitation, anonymisation, fairness, transparency, human in the loop.

4. **Expanded methodology**  
   - Four models: Logistic Regression, Random Forest, SVM, Gradient Boosting.  
   - Full preprocessing pipeline, stratified splits, cross-validation, and metrics (accuracy, precision, recall, F1, ROC-AUC).

5. **Implementation**  
   - `src/data_preprocessing.py`: load, clean, derive target, encode, scale, stratified split.  
   - `src/train_and_evaluate.py`: train all four models, evaluate, cross-validate, save models and results.  
   - `src/run_pipeline.py`: one command to run the full pipeline.

---

## Phase 2 Document Structure (Mandatory)

The document in `docs/Phase2_Document_Jariwala_Sakshi_4243407.md` follows the required structure:

- Title and Abstract  
- Introduction and Background  
- Objectives  
- Methodology  
- Timeline and Milestones  
- System Design and Architecture  
- Expected Challenges and Solutions  
- Conclusion  
- Appendix: Machine Learning Canvas (one-pager)

---

## How to Run and Check That Everything Works

### Step 1: Open a terminal in the project folder

In VS Code/Cursor: **Terminal → New Terminal**, or open PowerShell/Command Prompt and go to:

```text
F:\ALL IU Assignments\Project computer science\student-performance-ml
```

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

- **Console:** “Loading data…” → “Preprocessing…” → “Training and evaluating…” → a **MODEL COMPARISON** table with Accuracy, Precision, Recall, F1, ROC-AUC and Confusion Matrix for each of the 4 models, then **5-Fold Stratified CV (F1)** summary, and finally “Results and models saved to: …\output”.
- **No Python errors** (exit code 0). If you see a PowerShell “Get-ChildItem” message at the very end, you can ignore it; it’s a shell quirk, not your code.

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

## File Naming for Submission

- **Phase 2 document (PebblePad):**  
  `Jariwala_Sakshi_4243407_UseCaseAnalysis_P2_S` (export the Markdown/Word document as PDF).

- **Zip folder (Phase 3):**  
  `Jariwala_Sakshi_4243407` (or as specified), with subfolders e.g. `01-Research-and-Development`, `02-Conception`, `03-Finalization`.

---

## Next Steps for You

1. **Copy the Phase 2 document** into Word (or your editor), adjust wording if needed, and **export as PDF** for submission.  
2. **Export the Mermaid diagrams** from `docs/phase2_workflow_diagram.md` as images and paste them into the PDF (Section 5).  
3. **Run the pipeline** at least once and, if you wish, add a short “Results” subsection with a table or screenshot from `evaluation_results.json`.  
4. **Draw or digitise the Machine Learning Canvas** as a one-page figure and add it at the end of the Phase 2 document.  
5. **Phase 3:** Incorporate tutor feedback, add the 5-slide pitch deck and 2-page abstract, and zip all files as required.
