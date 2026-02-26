import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data_iris():
    """
    Load the Iris dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): The features of the Iris dataset.
        y (numpy.ndarray): The target values of the Iris dataset.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def load_data():
    """
    Load the raw financial dataset. Feature engineering is delegated to src.features.
    """
    df = pd.read_csv("data/final_data.csv")
    return df
    
def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    return X_train, X_test, y_train, y_test