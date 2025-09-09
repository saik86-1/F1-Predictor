import pandas as pd

# load cleaned dataset (already in true chronological order)
df = pd.read_csv("data/F1_RaceResult_Clean.csv")

# drop DNFs/DNS rows (Final Position must be numeric for rolling stats)
df = df.dropna(subset=["Final Position"]).copy()

# --- driver form features (respect existing row order) ---
# rolling over the *previous* races only (via shift), window=5 with min_periods=1
df["Driver_AvgFinish_Last5"] = (
    df.groupby("Driver Name", sort=False)["Final Position"]
      .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
)

df["Driver_PodiumRate_Last5"] = (
    df.groupby("Driver Name", sort=False)["Podium"]
      .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
)

df["Driver_PrevFinish"] = (
    df.groupby("Driver Name", sort=False)["Final Position"].shift()
)

# quick preview
print(df[["Year","Circuit Name","Driver Name",
          "Driver_AvgFinish_Last5","Driver_PodiumRate_Last5","Driver_PrevFinish","Podium"]].tail(15))

# save without changing order
df.to_csv("data/F1_RaceResult_DriverFeatures.csv", index=False)
print("✅ Saved driver feature dataset with shape:", df.shape)
