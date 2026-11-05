# 🚀 GitHub Actions Workflows

[![▶️ Run Pipeline & CI Workflow](https://img.shields.io/badge/▶%EF%B8%8F%20RUN%20WORKFLOW-CLICK%20HERE%20TO%20TRIGGER-2ea44f?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/mightyalok00/apple-review-sentiment-lstm/actions/workflows/ci.yml)
[![Live Streamlit Dashboard](https://img.shields.io/badge/🌐%20STREAMLIT%20APP-RUN%20IN%20BROWSER-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://apple-review-sentiment-lstm.streamlit.app/)
[![Workflow Status](https://github.com/mightyalok00/apple-review-sentiment-lstm/actions/workflows/ci.yml/badge.svg)](https://github.com/mightyalok00/apple-review-sentiment-lstm/actions/workflows/ci.yml)

---

## 🔘 How to Run the Workflow

Click the button above or follow these direct steps to manually execute the end-to-end pipeline on GitHub Actions:

### Option 1: Direct 1-Click Trigger

👉 **[Launch Pipeline & CI Workflow in GitHub Actions](https://github.com/mightyalok00/apple-review-sentiment-lstm/actions/workflows/ci.yml)**

1. On the workflow page, locate the **"Run workflow"** dropdown button on the right side.
2. Select branch `main`.
3. Choose the pipeline stages to run:
   - `partition,parquet,test` (Standard Fast CI - Recommended)
   - `all` (Full 5-Stage Pipeline including web scraper and neural retraining)
   - `partition,parquet` (Data engineering validation only)
   - `test` (Pytest 18-test suite execution only)
4. Click the green **"Run workflow"** button to start execution.

---

### Option 2: Run via Streamlit Web Dashboard

You can also trigger workflow stages interactively from the live application:

1. Open the **[Apple Sentiment Streamlit App](https://apple-review-sentiment-lstm.streamlit.app/)**.
2. Select **Tab 5: 🔄 Pipeline Workflow & Run**.
3. Choose your desired stages and hyperparameters.
4. Click **▶️ Run Full Workflow** or any of the one-click quick preset buttons.

---

### Option 3: Run Locally via Command Line

```bash
# Run data validation, Parquet sync, and 18-test pytest suite
python workflow.py --stages partition,parquet,test

# Run all 5 stages headlessly
python workflow.py --all

# Using Makefile target
make workflow-run
```

---

## 📁 Workflow Configuration Files

| File | Purpose | Trigger Events |
| :--- | :--- | :--- |
| [`ci.yml`](file:///c:/Users/Alok%20Agarwal/OneDrive/Desktop/alokkkk-main/.github/workflows/ci.yml) | End-to-end testing, syntax verification, model evaluation benchmarks, reproducibility manifest, and pipeline orchestration | `push`, `pull_request`, `workflow_dispatch` (Manual) |
