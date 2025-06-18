import pandas as pd
import joblib

from Preprocess import clean_data

def predict_new(input_data: pd.DataFrame):
    # Load Model
    model = joblib.load("models/severity_model.pkl")

    # Data Cleaning
    X, _ = clean_data(input_data)

    # Predict
    predictions = model.predict(X)
    return predictions
