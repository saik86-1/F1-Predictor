import pandas as pd

df = pd.read_csv("F1_RaceResult_2019_2025.csv")

# --- clean Starting Grid ---
# Convert numeric strings to int, keep "Pit Lane" as -1 (special flag)
df["Starting Grid"] = df["Starting Grid"].replace("Pit Lane", -1)
df["Starting Grid"] = pd.to_numeric(df["Starting Grid"], errors="coerce")

# --- clean Final Position ---
# Keep only numeric results, mark DNFs/DNS as NaN
df["Final Position"] = pd.to_numeric(df["Final Position"], errors="coerce")

# --- create target column (Podium: top 3) ---
df["Podium"] = df["Final Position"].apply(lambda x: 1 if pd.notnull(x) and x <= 3 else 0)

print(df[["Year", "Circuit Name", "Driver Name", "Starting Grid", "Final Position", "Podium"]].head(15))
print("\nClass balance (0=not podium, 1=podium):")
print(df["Podium"].value_counts())

# save a cleaned copy for modeling
df.to_csv("F1_RaceResult_Clean.csv", index=False)
