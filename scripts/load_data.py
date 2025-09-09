import pandas as pd

# Load CSV
df = pd.read_csv("F1_RaceResult_2019_2025.csv")

# Quick checks
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print(df.info())