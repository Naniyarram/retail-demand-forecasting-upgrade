# RetailCast — Retail Demand Forecasting & MLOps Platform

**End-to-end ML system: time-series forecasting → champion selection → API serving → inventory optimization → LLM business insights**

| | |
|---|---|
| **Dataset** | 421K+ Walmart weekly sales records · 45 stores · 81 departments |
| **Best model** | SARIMA · **3.95% MAPE** (5-fold walk-forward CV) |
| **Stack** | Python · MLflow · FastAPI · Docker · Hugging Face Llama 3.1 |
| **Quality** | 83 pytest cases · GitHub Actions CI/CD |

Built to demonstrate production-minded ML engineering — not a notebook-only exercise.

---

## What This Demonstrates (ML / MLOps / GenAI roles)

- **Applied ML** — Multi-model comparison (SARIMA, Prophet, XGBoost) with leakage-safe walk-forward validation and automated champion selection
- **MLOps** — MLflow experiment tracking, model packaging, FastAPI inference, drift monitoring, Docker deployment, CI pipeline
- **GenAI integration** — Hugging Face Llama 3.1 generates structured retail recommendations from forecast metrics, with output verification and safe fallback
- **Business impact** — Forecasts drive Safety Stock, Reorder Point, EOQ, and stockout/overstock risk scoring

---

## System Overview

```
Data (Walmart CSVs)
    → Feature engineering & aggregation
    → SARIMA / Prophet / XGBoost (walk-forward CV)
    → Champion selection (lowest MAPE) + MLflow logging
    → Model packaging (Joblib + metadata)
    → FastAPI serving + Streamlit dashboard
    → Inventory optimization + drift detection
    → LLM advisory layer (HF Inference Providers)
```

**Serving endpoints:** `/forecast` · `/inventory/optimize` · `/inventory/risk` · `/decision/recommendations` · `/monitoring/metrics`

---

## Results

| Rank | Model | RMSE | MAE | MAPE |
|---:|---|---:|---:|---:|
| 1 | **SARIMA** | $2,700,491 | $1,937,834 | **3.95%** |
| 2 | Prophet | $3,676,451 | $2,612,769 | 5.20% |
| 3 | XGBoost | $4,985,476 | $3,433,547 | 6.73% |

Champion config: `SARIMA(1,0,0)(0,0,2,52)` — retrained on full data and served in production.

---

## Tech Stack

| Area | Tools |
|---|---|
| Modeling | SARIMA (pmdarima), Prophet, XGBoost, scikit-learn |
| MLOps | MLflow, walk-forward CV, Joblib artifacts, model registry hooks |
| Serving | FastAPI, Uvicorn, Pydantic |
| GenAI | Hugging Face Inference Providers (`meta-llama/Llama-3.1-8B-Instruct`) |
| UI | Streamlit, Altair |
| Ops | Docker, Docker Compose, GitHub Actions, pytest |

---

## Quick Start

```bash
# 1. Setup
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. LLM insights (optional — copy and add your token)
cp .env.example .env

# 3. Train models & generate champion artifact
python run_experiments.py

# 4. Start API (terminal 1)
python run_api.py

# 5. Start dashboard (terminal 2)
streamlit run streamlit_app.py
```

**Docker (API only):**
```bash
docker compose up --build
```

**Verify:**
```bash
python verify_all.py
pytest
```

| Service | URL |
|---|---|
| API | http://127.0.0.1:8000 |
| API docs | http://127.0.0.1:8000/docs |
| Dashboard | http://localhost:8501 |

---

## Project Structure

```
pipeline/
  preprocessing/     # Data loading, aggregation, feature engineering
  forecasting/       # SARIMA, Prophet, XGBoost
  evaluation/        # Walk-forward CV, metrics
  training/          # Experiments, champion pipeline, MLflow
  api/               # FastAPI app & forecast service
  inventory/         # SS, ROP, EOQ, risk classification
  monitoring/        # KS-test drift detection
  utils/             # LLM client
streamlit_app.py     # Interactive dashboard
run_experiments.py   # Training entry point
run_api.py           # API entry point
tests/               # 19 test modules
```

---

## Resume-Ready Bullets

- Built an end-to-end retail demand forecasting platform on 421K+ weekly sales records (45 stores), achieving **3.95% MAPE** with walk-forward validated SARIMA, Prophet, and XGBoost and automated champion selection.
- Shipped a production-style MLOps pipeline: MLflow tracking, model packaging, FastAPI serving, drift monitoring, Docker deployment, and GitHub Actions CI (**83 tests**).
- Connected ML outputs to supply chain decisions via Safety Stock, Reorder Point, EOQ, and stockout/overstock risk classification.
- Integrated Hugging Face Llama 3.1 for structured business recommendations from forecast metrics, with output validation and deterministic fallback.

---

**Author:** Nani
