import pandas as pd

# Load dataset
df = pd.read_csv("data/Hotel Reservations.csv")

print("Dataset loaded successfully!\n")

print(df.head())

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBooking Status Count:")
print(df["booking_status"].value_counts())