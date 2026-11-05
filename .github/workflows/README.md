# 🚀 GitHub Actions Workflows

[![▶️ Run Pipeline & CI Workflow](https://img.shields.io/badge/▶%EF%B8%8F%20RUN%20WORKFLOW-CLICK%20HERE%20TO%20TRIGGER-2ea44f?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/mightyalok00/apple-review-sentiment-lstm/actions/workflows/ci.yml)
[![Live Streamlit Dashboard](https://img.shields.io/badge/🌐%20STREAMLIT%20APP-RUN%20IN%20BROWSER-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://apple-review-sentiment-lstm.streamlit.app/)
[![Workflow Status](https://github.com/mightyalok00/apple-review-sentiment-lstm/actions/workflows/ci.yml/badge.svg)](https://github.com/mightyalok00/apple-review-sentiment-lstm/actions/workflows/ci.yml)

---

## 🔘 How to Run the Workflow (Mobile & Desktop)

Click the button above or follow these direct steps to manually execute the end-to-end pipeline on GitHub Actions from either desktop or mobile:

### Option 1: Direct 1-Click Trigger (Mobile Web, GitHub Mobile App & Desktop)

👉 **[Launch Pipeline & CI Workflow in GitHub Actions](https://github.com/mightyalok00/apple-review-sentiment-lstm/actions/workflows/ci.yml)**

- **On Desktop**:
  1. Open the [Workflow Page](https://github.com/mightyalok00/apple-review-sentiment-lstm/actions/workflows/ci.yml).
  2. Click the **"Run workflow"** button on the top-right.
  3. Select branch `main`.
  4. Choose your stages:
     - `partition,parquet,test` (Standard Fast CI - Recommended)
     - `partition,parquet,test,eval` (Fast CI + Academic Model Evaluation)
     - `eval` (Standalone `evaluate_model.py` with 95% CIs, McNemar tests & calibration)
     - `all` (Full 6-Stage Pipeline: scrape, partition, parquet, train, test, eval)
     - `test` (Pytest 21-test suite execution only)
     - `partition,parquet` (Data engineering validation only)
  5. Click the green **"Run workflow"** button.

- **On Mobile (Browser or GitHub Mobile App)**:
  1. Open the link above in Chrome, Safari, or the GitHub Mobile App.
  2. Tap the **"Run workflow"** banner at the top of the workflow view.
  3. Select `main` and pick your preferred stage preset.
  4. Tap the green **"Run workflow"** button.

---

### Option 2: Run via Streamlit Web Dashboard (Mobile & Desktop Responsive)

You can also trigger workflow stages interactively from the live application with touch-friendly controls:

1. Open the **[Apple Sentiment Streamlit App](https://apple-review-sentiment-lstm.streamlit.app/)**.
2. **From Any Screen / Mobile Drawer**: Use the **⚡ Quick Pipeline Launcher** in the header or sidebar.
3. **From Tab 5 (Pipeline Workflow & Run)**:
   - Configure custom stages and hyperparameters.
   - Tap **▶️ Run Full Workflow**, **📦 Run Data Validation & Parquet**, **🧪 Run Automated Test Suite**, or **🔬 Run Baseline & Evaluation (`evaluate_model.py`)**.

---

### Option 3: Run Locally via Command Line

```bash
# Run data validation, Parquet sync, and 21-test pytest suite
python workflow.py --stages partition,parquet,test

# Run data validation, Parquet sync, tests, and academic model evaluation
python workflow.py --stages partition,parquet,test,eval

# Run standalone model evaluation suite directly
python evaluate_model.py --seed 42

# Run all 6 stages headlessly
python workflow.py --all

# Using Makefile target
make workflow-run
```

---

## 📁 Workflow Configuration Files

| File | Purpose | Trigger Events |
| :--- | :--- | :--- |
| [`ci.yml`](file:///c:/Users/Alok%20Agarwal/OneDrive/Desktop/alokkkk-main/.github/workflows/ci.yml) | End-to-end testing, syntax verification, model evaluation benchmarks, reproducibility manifest, and pipeline orchestration | `push`, `pull_request`, `workflow_dispatch` (Manual) |
