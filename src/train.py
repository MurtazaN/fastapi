from sklearn.tree import DecisionTreeClassifier
import joblib
from src.data import load_data, split_data
from xgboost import XGBClassifier
from src.features import engineer_target_and_features
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model"


def fit_model_iris(X_train, y_train):
    """
    Train a Decision Tree Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
    dt_classifier.fit(X_train, y_train)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(dt_classifier, MODEL_DIR / "iris_model.pkl")


def fit_model_financial(X_train, y_train):
    """
    Train a XGBoost Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    xgb_classifier = XGBClassifier(max_depth=3, random_state=12)
    xgb_classifier.fit(X_train, y_train)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(xgb_classifier, MODEL_DIR / "financial_model.pkl")

if __name__ == "__main__":
    # Load raw dataframe
    df_raw = load_data()
    # Sent to features.py to engineer targets and features (and save the fitted categorical encoders!)
    X, y = engineer_target_and_features(df_raw, is_training=True)
    # Split the processed data
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Send to fit_model_financial to train and save the XGBoost model
    fit_model_financial(X_train, y_train)
