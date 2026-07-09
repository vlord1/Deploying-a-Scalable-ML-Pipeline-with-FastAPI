import os
import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.model import compute_model_metrics, inference, train_model

@pytest.fixture
def dummy_data():
    """
    Fixture to provide basic synthetic data for testing model functions.
    """
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([0, 1, 0, 1])
    return X, y

def test_train_model_is_random_forest(dummy_data):
    """
    Test 1: Checks if the ML model uses the expected algorithm.
    """
    X, y = dummy_data
    model = train_model(X, y)
    
    # Assert that the returned model is indeed a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)

def test_inference_returns_numpy_array(dummy_data):
    """
    Test 2: Checks if the inference function returns the expected type of result.
    """
    X, y = dummy_data
    model = train_model(X, y)
    predictions = inference(model, X)
    
    # Assert that predictions is a numpy array
    assert isinstance(predictions, np.ndarray)
    # Assert that the number of predictions matches the number of input rows
    assert len(predictions) == len(X)

def test_compute_model_metrics_returns_valid_ranges():
    """
    Test 3: Checks if the computing metrics functions return expected value types and ranges.
    """
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # Assert that all metrics are floats
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    
    # Assert that all metrics fall within the standard probability bounds of 0.0 to 1.0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0

@pytest.fixture
def census_data():
    """
    Fixture to load the actual census dataset for data validation tests.
    """
    data_path = os.path.join(".", "data", "census.csv")
    return pd.read_csv(data_path)

def test_data_no_null_categorical_values(census_data):
    """
    Test 4: Checks if the data meets expectations by validating that required 
    categorical columns have no missing values.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    # Assert that all required categorical columns exist in the dataset
    for col in cat_features:
        assert col in census_data.columns, f"Required column '{col}' is missing."
        
    # Assert that there are no missing/null values in these columns
    for col in cat_features:
        null_count = census_data[col].isnull().sum()
        assert null_count == 0, f"Column '{col}' contains {null_count} missing values."