from pydantic import BaseModel
from typing import Optional

class FinancialRequest(BaseModel):
    """
    Pydantic BaseModel representing the required input features for
    predicting a user's financial condition using XGBoost.
    
    Excludes the target columns and identifiers.
    """
    age: int
    gender: str
    education_level: str
    employment_status: str
    job_title: str
    monthly_income_usd: float
    monthly_expenses_usd: float
    savings_usd: float
    has_loan: str
    loan_type: Optional[str] = "None"
    loan_amount_usd: float
    loan_term_months: int
    monthly_emi_usd: float
    loan_interest_rate_pct: float
    region: str


class FinancialResponse(BaseModel):
    """
    Pydantic BaseModel formatting the XGBoost prediction output.
    """
    prediction: int  # 1 for Good Financial Condition, 0 for otherwise
    status_label: str # "Good" or "Needs Improvement"
