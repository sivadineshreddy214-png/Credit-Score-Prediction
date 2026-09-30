from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# CREATE FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="Customer Credit Risk API",
    description="FastAPI backend for Customer Credit Risk Prediction",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

# main.py is in App/backend, model files are in App/pkl.files
BASE_DIR = Path(__file__).resolve().parent
PKL_DIR = BASE_DIR / "models"

MODEL_PATH = PKL_DIR / "customer_credit_risk_model.pkl"
IMPUTER_PATH = PKL_DIR / "credit_risk_imputer.pkl"
FEATURES_PATH = PKL_DIR / "credit_risk_features.pkl"


def load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return joblib.load(path)


model = load_artifact(MODEL_PATH)
imputer = load_artifact(IMPUTER_PATH)
feature_names = list(load_artifact(FEATURES_PATH))


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class CreditRiskRequest(BaseModel):

    Monthly_Inhand_Salary: float = Field(
        ..., description="Monthly income after deductions"
    )

    Total_EMI_per_month: float = Field(
        ..., description="Total EMI paid per month"
    )

    Outstanding_Debt: float = Field(
        ..., description="Outstanding debt"
    )

    Credit_Utilization_Ratio: float = Field(
        ..., description="Credit utilization ratio"
    )

    Num_of_Delayed_Payment: float = Field(
        ..., description="Number of delayed payments"
    )

    Monthly_Balance: float = Field(
        ..., description="Average monthly balance"
    )

    Num_Credit_Inquiries: float = Field(
        ..., description="Number of credit inquiries"
    )

    EMI_to_Income: float = Field(
        ..., description="EMI to income ratio"
    )


# --------------------------------------------------
# RESPONSE
# --------------------------------------------------

class CreditRiskResponse(BaseModel):

    prediction: int
    probabilities: dict


# --------------------------------------------------
# HOME API
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Customer Credit Risk API is running",
        "status": "success"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "imputer_loaded": imputer is not None
    }


# --------------------------------------------------
# CREDIT RISK PREDICTION
# --------------------------------------------------

@app.post("/predict", response_model=CreditRiskResponse)
def predict_credit_risk(data: CreditRiskRequest):

    # Convert request data into dictionary
    input_data = {
        "Monthly_Inhand_Salary": data.Monthly_Inhand_Salary,
        "Total_EMI_per_month": data.Total_EMI_per_month,
        "Outstanding_Debt": data.Outstanding_Debt,
        "Credit_Utilization_Ratio": data.Credit_Utilization_Ratio,
        "Num_of_Delayed_Payment": data.Num_of_Delayed_Payment,
        "Monthly_Balance": data.Monthly_Balance,
        "Num_Credit_Inquiries": data.Num_Credit_Inquiries,
        "EMI_to_Income": data.EMI_to_Income
    }

    # Create DataFrame in the exact feature order
    # Build the DataFrame in the exact feature order used in training
    missing = [f for f in feature_names if f not in input_data]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Model expects features not provided by the API: {missing}"
        )

    df = pd.DataFrame([input_data])[feature_names]

    # --------------------------------------------------
    # IMPUTATION
    # --------------------------------------------------

    # Imputer was trained on the first 7 features.
    imputer_features = [
        "Monthly_Inhand_Salary",
        "Total_EMI_per_month",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Num_of_Delayed_Payment",
        "Monthly_Balance",
        "Num_Credit_Inquiries"
    ]

    df[imputer_features] = imputer.transform(
        df[imputer_features]
    )

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    prediction = model.predict(df)[0]

    # --------------------------------------------------
    # PREDICTION PROBABILITY
    # --------------------------------------------------

    probabilities_array = model.predict_proba(df)[0]

    probabilities = {
        str(cls): round(float(prob), 4)
        for cls, prob in zip(
            model.classes_,
            probabilities_array
        )
    }

    return {
        "prediction": int(prediction),
        "probabilities": probabilities
    }


