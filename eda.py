import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Hotel Reservations.csv")

# Show first 5 rows
print(df.head())

# Show dataset info
print(df.info())

# Missing values
print(df.isnull().sum())

# Booking status counts
print(df['booking_status'].value_counts())

# Graph
sns.countplot(data=df, x='booking_status')
plt.title("Booking Status")
plt.show()