import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import joblib

def engineer_target_and_features(df: pd.DataFrame, is_training: bool = True):
    """
    Takes the raw financial dataframe and outputs the cleaned Feature Matrix (X)
    and Target Vector (y) ready for ML training or inference.
    """
    if is_training:
        # 1. Engineer Composite Target Variable (y)
        good_financial_condition = (
            (df["credit_score"] >= 700) & 
            (df["savings_to_income_ratio"] > 3.5) & 
            (df["debt_to_income_ratio"] < 3.0)
        ).astype(int)
        y = good_financial_condition
    else:
        y = None # No target during inference
        
    # 2. Select Features (X)
    columns_to_drop = [
        "user_id", 
        "record_date", 
        "credit_score", 
        "savings_to_income_ratio", 
        "debt_to_income_ratio"
    ]
    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

    # 3. Handle Categorical Features
    categorical_cols = ["gender", "education_level", "employment_status", "job_title", "has_loan", "loan_type", "region"]
    existing_cat_cols = [col for col in categorical_cols if col in X.columns]
    
    if is_training:
        # Fit global encoder and save it
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[existing_cat_cols] = encoder.fit_transform(X[existing_cat_cols])
        joblib.dump(encoder, "model/categorical_encoder.pkl")
    else:
        # Load fitted encoder during inference
        encoder = joblib.load("model/categorical_encoder.pkl")
        X[existing_cat_cols] = encoder.transform(X[existing_cat_cols])

    return X, y

def preprocess_inference_data(financial_request_dict: dict) -> np.ndarray:
    """
    Converts incoming Pydantic API payload into XGBoost format.
    """
    df = pd.DataFrame([financial_request_dict])
    X, _ = engineer_target_and_features(df, is_training=False)
    return X.values

