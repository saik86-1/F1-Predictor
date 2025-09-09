import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# load engineered data
df = pd.read_csv("data/F1_RaceResult_DriverFeatures.csv")

# keep only rows with features
df = df.dropna(subset=["Driver_AvgFinish_Last5","Driver_PodiumRate_Last5","Driver_PrevFinish"])

# train/test split (same as before)
X = df[["Driver_AvgFinish_Last5","Driver_PodiumRate_Last5","Driver_PrevFinish"]]
y = df["Podium"]

X_train = X[df["Year"] < 2025]
y_train = y[df["Year"] < 2025]

# train model (or load if you saved it earlier)
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# === predict for upcoming Baku 2025 ===

# get the last available race per driver (max Year + Circuit before Baku)
latest_rows = df[df["Year"] == 2025].groupby("Driver Name").tail(1)

X_pred = latest_rows[["Driver_AvgFinish_Last5","Driver_PodiumRate_Last5","Driver_PrevFinish"]]
probs = model.predict_proba(X_pred)[:,1]

latest_rows = latest_rows.copy()
latest_rows["Podium_Prob"] = probs

# rank by probability
predictions = latest_rows.sort_values("Podium_Prob", ascending=False)[
    ["Driver Name","Team","Driver_AvgFinish_Last5","Driver_PodiumRate_Last5","Driver_PrevFinish","Podium_Prob"]
]

print("=== Predicted podium contenders for Baku 2025 ===")
print(predictions.head(10))  # top 10 for context
