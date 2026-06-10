"""
Week 2/3 — Data Preprocessing
Hotel Reservation Cancellation Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pickle

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Hotel_Reservations.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame):
    """
    Clean, encode, scale, and split the dataset.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    df = df.copy()

    # Drop identifier
    df.drop(columns=["Booking_ID"], inplace=True)

    # Encode target
    df["booking_status"] = (df["booking_status"] == "Canceled").astype(int)

    # Encode categorical features
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    feature_names = [c for c in df.columns if c != "booking_status"]
    X = df[feature_names].values
    y = df["booking_status"].values

    # Train/test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Persist scaler and encoders
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    print(f"[INFO] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    print(f"[INFO] Cancellation rate — Train: {y_train.mean()*100:.1f}%  |  Test: {y_test.mean()*100:.1f}%")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    df = load_data()
    preprocess(df)
