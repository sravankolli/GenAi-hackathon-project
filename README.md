Project scaffold and objective
This gives you a complete, minimal, reproducible setup: a clear README, a training notebook-equivalent script, a tiny FastAPI service (no UI), dependencies, run commands, evaluation notes, and a practical commit workflow with AI chat log capture. It’s lean, hackathon-ready, and easy to extend.

Step-by-step setup
- Clone or create repo:
Create a new repo and copy the files below into the structure shown.
- Create virtual environment:
Use venv or conda and install dependencies from requirements.txt or environment.yml.
- Run training:
Train and save a model with the provided script.
- Start service:
Launch the FastAPI server and hit the predict endpoint.
- Evaluate:
Review metrics output and limitations, then iterate.

Code files
Repository layout
genai-pipeline/
├─ README.md
├─ requirements.txt
├─ environment.yml
├─ data/
│  └─ raw.csv            # placeholder, or downloaded via script
├─ models/
│  └─ model.pkl          # created by training
├─ app/
│  └─ main.py            # FastAPI service
├─ notebooks/
│  └─ 01_data_and_training.py  # reproducible "notebook-like" script
├─ tests/
│  └─ test_app.py        # minimal tests
└─ docs/
   ├─ evaluation_notes.md
   └─ ai_logs.md


README.md
# Real-time GenAI pipeline

## Problem statement
The goal is to build a minimal, reproducible pipeline that ingests tabular data, trains a baseline model, and exposes a tiny FastAPI prediction service. This acts as a foundation you can evolve into real-time GenAI use cases (e.g., feature engineering, embeddings, or re-ranking).

## Data link
Source: <PUT_YOUR_DATA_URL_HERE>  
Format: CSV (rows = samples, columns = features, target = "label").  
If no public link exists, place `raw.csv` in `data/`.

## Design
- Ingestion: Load CSV from `data/raw.csv` or download via script.
- Training: Scikit-learn Pipeline with preprocessing and a simple classifier.
- Persistence: Save fitted pipeline to `models/model.pkl`.
- Serving: FastAPI exposes `/health` and `/predict`.
- Reproducibility: Single script in `notebooks/01_data_and_training.py` with deterministic seeds.

## Assumptions
- Data includes a column named `label` as the target.
- Mixed numeric and categorical features are possible.
- No missing values beyond simple imputation needed.
- Baseline metrics are sufficient for initial evaluation; further tuning is out of scope.

## How to run
1. Create environment:


python -m venv .venv .venv\Scripts\activate  # Windows pip install -r requirements.txt
Or:


conda env create -f environment.yml conda activate genai-pipeline

2. Train the model and output metrics:


python notebooks/01_data_and_training.py --data data/raw.csv --model models/model.pkl

3. Start the API:


uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

4. Predict:


curl -X POST http://localhost:8000/predict 
-H "Content-Type: application/json" 
-d '{"features": {"feature1": 1.2, "feature2": "A"}}'

## Evaluation notes
See `docs/evaluation_notes.md` for metrics, tests, guardrails, and limitations.

## Commit history and AI chat logs
- Keep small, descriptive commits (see example commands below).
- Export AI chat logs to `docs/ai_logs.md` and reference them from PRs or the README.


notebooks/01_data_and_training.py
import argparse
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("Target column 'label' not found in dataset.")
    return df

def split_features_target(df: pd.DataFrame):
    y = df["label"]
    X = df.drop(columns=["label"])
    return X, y

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    return model

def train_and_eval(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_pipeline(X)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": list(np.unique(y))
    }
    return model, metrics

def save_model(model: Pipeline, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--model", type=str, default="models/model.pkl", help="Model output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seeds(args.seed)
    df = load_data(args.data)
    X, y = split_features_target(df)
    model, metrics = train_and_eval(X, y)
    save_model(model, args.model)

    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"Model saved to: {args.model}")

if __name__ == "__main__":
    main()


app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

app = FastAPI(title="GenAI Minimal Service", version="0.1.0")

class FeaturesPayload(BaseModel):
    features: dict

# Lazy load to start fast even if model missing; load when first used
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: FeaturesPayload):
    try:
        model = get_model()
        X = [payload.features]  # single sample
        pred = model.predict(X)[0]
        return {"prediction": str(pred)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")


tests/test_app.py
import os
import json
from fastapi.testclient import TestClient
from app.main import app, MODEL_PATH

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_requires_model():
    # Ensure missing model yields 404
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    payload = {"features": {"feature1": 1.0}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 404



Dependencies and run commands
requirements.txt
fastapi==0.115.2
uvicorn==0.30.0
pydantic==2.8.2
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
joblib==1.4.2
pytest==8.2.1


environment.yml
name: genai-pipeline
channels:
  - conda-forge
dependencies:
  - python=3.11
  - fastapi
  - uvicorn
  - pydantic
  - pandas
  - numpy
  - scikit-learn
  - joblib
  - pytest


Run commands
- Create and activate venv:
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
- Train model:
python notebooks/01_data_and_training.py --data data/raw.csv --model models/model.pkl
- Start API:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
- Run tests:
pytest -q



Evaluation notes and guardrails
- Metrics:
Accuracy, weighted F1, precision, recall are printed by the training script. Use a held-out test split with stratification to avoid class imbalance artifacts.
- Tests:
Basic health check and missing model behavior are covered. Extend with schema validation tests, sample predictions with a known input, and regression tests by pinning a fixture dataset and expected outputs.
- Guardrails:
- Input validation: Pydantic schema enforces a dict under features. Reject unknown types; prefer numeric or simple categorical values.
- Size limits: Add request size limits and feature count checks to prevent abuse.
- Error handling: Distinguish 404 when model is missing and 400 for bad requests.
- Reproducibility: Fixed random seeds, pinned dependencies, and commit small changes frequently.
- Data hygiene: Ensure no PII in logs; avoid storing raw requests.
- Limitations:
Baseline model only, no hyperparameter tuning, no feature selection, and simplistic preprocessing. FastAPI service is single-process and not optimized for high throughput. Real-time streaming, authentication, and monitoring are not included.

Commit history and ai chat logs
- Suggested commit workflow:
- Init repo:
git init
git add .
git commit -m "chore: initialize repo with scaffold and README"
- Add training script and data:
git add notebooks/01_data_and_training.py data/raw.csv
git commit -m "feat: add reproducible training script and sample data"
- Add FastAPI service:
git add app/main.py
git commit -m "feat(api): minimal FastAPI predict endpoint"
- Add tests and docs:
git add tests/test_app.py docs/evaluation_notes.md docs/ai_logs.md
git commit -m "test: add basic API tests; docs: evaluation notes and AI logs"
- Record metrics and model artifact:
git add models/model.pkl
git commit -m "build: add trained model artifact for reproducibility"
- AI chat logs:
- Export: Copy relevant chat transcripts to docs/ai_logs.md.
- Reference: Link this file in PR descriptions or README for transparency.
- Example header in ai_logs.md:
# AI chat logs
- Date: 2025-12-12
- Context: Model design decisions and API choices
- Summary: Baseline logistic regression with preprocessing; FastAPI for serving
- Transcript: <Paste here or link to exported chat>



If you share your data schema or target task, I’ll tailor the preprocessing, metrics, and endpoint schema to fit your exact use case.
