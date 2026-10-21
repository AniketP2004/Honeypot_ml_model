# Honeypot IDS: ML-Powered Intrusion Detection

A full-stack intrusion detection system that classifies attacker behavior captured by a honeypot. Incoming requests are converted to features, scored by a trained ML model, and logged to a live dashboard.

**Live demo:** https://honeypot-ml-model-6.onrender.com/

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-database-336791)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)

---

## What It Does

A honeypot is a decoy system built to attract attackers. This project goes one step further: every captured interaction is classified in real time into one of four behavior classes.

| Class | Description |
|---|---|
| `automated_scan` | Bot-driven probing and port/service scanning |
| `brute_force` | Repeated credential guessing |
| `web_recon` | Web reconnaissance and path enumeration |
| `normal` | Benign traffic |

---

## Features

- **Real-time classification** of requests via a FastAPI endpoint.
- **Three trained models**: GradientBoosting (best), RandomForest, XGBoost.
- **Model registry** (`ml_registry.py`) that loads all serialized models with joblib.
- **Shared feature pipeline** (`features.py`) built on the same extraction functions used in training (`data_cleaning.py`), so training and serving stay consistent.
- **Auto-retrain watchdog**: monitors incoming logs and triggers retraining with a 30-second debounce. Thread-safe, with logging and byte-position tracking.
- **Attack log dashboard** backed by PostgreSQL + SQLAlchemy.
- **Dark, cyber-themed frontend** in Vanilla JS with five sections: Hero/Team, How It Works, Live Demo, Attack Logs, Research Papers.
- **Demo override**: `request_rate_override` field for testing high-rate scenarios without generating real traffic.

---

## Model Performance

| Model | Notes |
|---|---|
| **GradientBoosting** | Best performer, **F1 = 0.9979** |
| RandomForest | Strong baseline |
| XGBoost | Comparable alternative |

### Dataset

- **2,369 samples** after merging external datasets (SSH Logs, AWS Honeypot, Botnet SSH from Kaggle) with 100 synthetic `exploit_attempt` rows.
- Preprocessing used an 8-class taxonomy, later merged into 4 classes for serving.

### Engineering notes

Issues found and fixed during development:

- Data leakage from `balance_classes()` running before the train/test split.
- `TargetEncoder` silently defaulting to the majority class on high-cardinality strings, so it was removed entirely.
- Stale duplicate `.pkl` files in two locations.
- Label encoder mismatches between training and serving.

Only scikit-learn-native models are retained in serving.

---

## Architecture

```
 Honeypot traffic / demo input
            │
            ▼
   ┌─────────────────┐
   │   FastAPI API   │
   └───┬─────────┬───┘
       │         │
       ▼         ▼
 features.py  PostgreSQL ◀── attack logs
       │                          ▲
       ▼                          │
  ml_registry.py ── model ── prediction
       ▲
       │ retrain
  Watchdog (30s debounce)
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | FastAPI, SQLAlchemy |
| Database | PostgreSQL |
| ML | scikit-learn, XGBoost, joblib |
| Frontend | Vanilla JS |
| Deployment | Render |

---

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL

### Install

```bash
git clone https://github.com/AniketP2004/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/honeypot
```

### Run

```bash
uvicorn app.main:app --reload
```

API docs: `http://localhost:8000/docs`

Tables are created automatically on startup.

---

## Deployment Notes

Deployed on **Render**. On the free tier, the managed PostgreSQL instance expires 30 days after creation. A startup handler recreates the tables automatically, so the app recovers with a fresh database.

---

## Team

| Name | Role |
|---|---|
| **Aniket** ([@AniketP2004](https://github.com/AniketP2004)) | ML Engineer |
| Aakash Wavhal | Backend & Honeypot Infrastructure Lead |
| Vaishnavi Chavan | Backend & Honeypot Infrastructure Lead |
| Yuvraj Dudhal | Adversarial Testing Lead |
