from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
import os

# =========================================
# CONFIG
# =========================================

MODEL_PATH = "./best_pipeline_xgb.joblib"

# =========================================
# FASTAPI
# =========================================

app = FastAPI(title="Marketing Budget Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# LOAD MODEL
# =========================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully")

# =========================================
# INPUT SCHEMA
# =========================================

class PredictInput(BaseModel):
    data: Dict[str, Any]

# =========================================
# NORMALIZATION
# =========================================

def normalize_multilabel(value):

    if pd.isna(value):
        return ""

    value = (
        str(value)
        .replace(";", "|")
        .replace(",", "|")
        .replace("/", "|")
    )

    parts = [
        p.strip().lower()
        for p in value.split("|")
        if p.strip()
    ]

    return "|".join(sorted(set(parts)))

# =========================================
# ROUTES
# =========================================

@app.get("/")
def home():
    return {"message": "API Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# =========================================
# PREDICT
# =========================================

@app.post("/predict")
def predict(input_data: PredictInput):

    try:

        data = input_data.data

        # =========================
        # DATAFRAME
        # =========================

        df = pd.DataFrame([data])

        # =========================
        # NORMALIZE MULTI LABELS
        # =========================

        multilabel_cols = [
            "channels",
            "keyDeliverables",
            "metrics"
        ]

        for col in multilabel_cols:
            if col in df.columns:
                df[col] = df[col].apply(normalize_multilabel)

        # =========================
        # EXPECTED COLS
        # =========================

        expected_cols = []

        try:

            preproc = pipeline.named_steps.get("preproc")

            if preproc is not None:

                for name, transformer, cols in preproc.transformers_:

                    if cols is not None:
                        expected_cols += list(cols)

        except Exception:
            expected_cols = []

        # =========================
        # ALIGN COLUMNS
        # =========================

        if expected_cols:

            for c in expected_cols:

                if c not in df.columns:

                    if c in [
                        "estimatedDurationDays",
                        "teamSize",
                        "mediaBudget",
                        "externalCosts",
                        "fixedCosts",
                        "contingencyPercent"
                    ]:
                        df[c] = 0.0
                    else:
                        df[c] = "missing"

            X = df[expected_cols]

        else:
            X = df

        # =========================
        # PREDICTION
        # =========================

        prediction = pipeline.predict(X)[0]

        return {
            "predictedBudget": float(prediction)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )