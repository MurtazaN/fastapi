import joblib
from src.features import preprocess_inference_data

def predict_data_iris(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = joblib.load("../model/iris_model.pkl")
    y_pred = model.predict(X)
    return y_pred

def predict_data_financial(financial_request_dict: dict):
    """
    Predict the financial condition for the input API payload.
    Args:
        financial_request_dict (dict): Input data dict from Pydantic model.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    # 1. Preprocess the raw dictionary into an XGBoost-ready numpy array
    X = preprocess_inference_data(financial_request_dict)
    
    # 2. Load model and predict
    model = joblib.load("model/financial_model.pkl")
    y_pred = model.predict(X)
    return y_pred
    

