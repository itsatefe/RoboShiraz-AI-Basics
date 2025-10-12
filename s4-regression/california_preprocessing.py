"""
Preprocessing module for the California Housing Dataset
-------------------------------------------------------
This script includes the preprocessing steps necessary to prepare the California housing data
for machine learning tasks. Each step is commented and justified.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

def load_data():
    """
    Loads the California housing dataset and returns it as a pandas DataFrame.
    """
    cali = fetch_california_housing(as_frame=True)
    X = cali.data
    y = cali.target
    return X, y

def preprocess_data(X, y):
    """
    Applies preprocessing to the feature matrix X.
    Steps:
    1. Handle missing values
    2. Feature scaling using StandardScaler
    """
    # Step 1: Handle missing values
    # Although California housing data doesn't have missing values, we use SimpleImputer
    # to ensure the pipeline is robust to future changes or different datasets.
    imputer = SimpleImputer(strategy="median")

    # Step 2: Feature scaling
    # StandardScaler standardizes features by removing the mean and scaling to unit variance.
    # This is essential for many ML models like linear regression, SVMs, and neural networks.
    scaler = StandardScaler()

    # Create a pipeline to apply both transformations in sequence
    pipeline = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler)
    ])

    # Fit and transform the features
    X_prepared = pipeline.fit_transform(X)

    return X_prepared, y


