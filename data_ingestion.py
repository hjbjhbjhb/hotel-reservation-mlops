"""
Week 2 — Data Ingestion & EDA
Hotel Reservation Cancellation Prediction
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Hotel_Reservations.csv")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def basic_eda(df: pd.DataFrame) -> None:
    print("\n===== BASIC EDA =====")
    print("\n-- Shape --")
    print(df.shape)

    print("\n-- Data Types --")
    print(df.dtypes)

    print("\n-- Missing Values --")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")

    print("\n-- Target Distribution --")
    counts = df["booking_status"].value_counts()
    pcts = df["booking_status"].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"count": counts, "percent": pcts.round(2)})
    print(summary)

    print("\n-- Numerical Summary --")
    print(df.describe().T)

    print("\n-- Categorical Columns --")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        if col not in ("Booking_ID", "booking_status"):
            print(f"\n  {col}:")
            print(df[col].value_counts().to_string())


def key_observations(df: pd.DataFrame) -> None:
    print("\n===== KEY OBSERVATIONS =====")

    canceled = (df["booking_status"] == "Canceled").mean() * 100
    print(f"  Cancellation rate: {canceled:.1f}%")

    avg_lead_canceled = df[df["booking_status"] == "Canceled"]["lead_time"].mean()
    avg_lead_kept = df[df["booking_status"] == "Not_Canceled"]["lead_time"].mean()
    print(f"  Avg lead time (Canceled):     {avg_lead_canceled:.1f} days")
    print(f"  Avg lead time (Not_Canceled): {avg_lead_kept:.1f} days")

    avg_price_canceled = df[df["booking_status"] == "Canceled"]["avg_price_per_room"].mean()
    avg_price_kept = df[df["booking_status"] == "Not_Canceled"]["avg_price_per_room"].mean()
    print(f"  Avg room price (Canceled):     ${avg_price_canceled:.2f}")
    print(f"  Avg room price (Not_Canceled): ${avg_price_kept:.2f}")

    special_cancel = df[df["booking_status"] == "Canceled"]["no_of_special_requests"].mean()
    special_kept = df[df["booking_status"] == "Not_Canceled"]["no_of_special_requests"].mean()
    print(f"  Avg special requests (Canceled):     {special_cancel:.2f}")
    print(f"  Avg special requests (Not_Canceled): {special_kept:.2f}")


if __name__ == "__main__":
    df = load_data()
    basic_eda(df)
    key_observations(df)
