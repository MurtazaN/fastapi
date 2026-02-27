from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.features import engineer_target_and_features


BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model"


def main():
    '''
    # Instructions to test the accuracy of the model:
    1. Activate the environment (if not already activated)
    source fastapi_env/bin/activate
    2. Train the model (if not already trained)
    python3 src/train.py
    3. Test the accuracy of the model
    python3 -m src.test_accuracy
    '''
    # Load raw data
    df = load_data()

    # Recreate the engineered target exactly (same as src/features.py)
    y = (
        (df["credit_score"] >= 700)
        & (df["savings_to_income_ratio"] > 3.5)
        & (df["debt_to_income_ratio"] < 3.0)
    ).astype(int)

    # Build X using inference preprocessing (loads saved encoder)
    X, _ = engineer_target_and_features(df, is_training=False)

    # Split (match your src/data.py defaults)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12
    )

    # Load trained model + evaluate
    model = joblib.load(MODEL_DIR / "financial_model.pkl")
    y_pred = model.predict(X_test)

    print("--------------------------------")
    print("accuracy:", accuracy_score(y_test, y_pred))
    print("--------------------------------")
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("--------------------------------")
    print("classification report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()