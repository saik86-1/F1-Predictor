import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, classification_report

# ---- load & prep ----
df = pd.read_csv("data/F1_RaceResult_DriverFeatures.csv")
df = df.dropna(subset=["Driver_AvgFinish_Last5","Driver_PodiumRate_Last5","Driver_PrevFinish"])
X = df[["Driver_AvgFinish_Last5","Driver_PodiumRate_Last5","Driver_PrevFinish"]]
y = df["Podium"]

# ---- custom time split ----
# Test: only 2025 Italy & Netherlands; Train: everything else (incl. other 2025 races and all <= 2024)
test_races = ["Italy", "Netherlands"]
test_mask  = (df["Year"] == 2025) & (df["Circuit Name"].isin(test_races))
train_mask = ~test_mask  # all remaining rows

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_test,  y_test  = X.loc[test_mask],  y.loc[test_mask]

# ---- model ----
model = XGBClassifier(
    n_estimators=125,
    learning_rate=0.02,
    max_depth=3,
    subsample=0.6,
    colsample_bytree=0.5,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum()
)
model.fit(X_train, y_train)

# ---- evaluate on TEST (2025 Italy + Netherlands) ----
y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)[:, 1]
mae_test = mean_absolute_error(y_test, y_prob_test)

print("=== Classification Report (Test: 2025 Italy + Netherlands) ===")
print(classification_report(y_test, y_pred_test, digits=3))

# ---- predict top-3 podium contenders (using each driver's latest 2025 row as current form) ----
latest_2025 = df[df["Year"] == 2025].groupby("Driver Name", as_index=False).tail(1).copy()
X_pred = latest_2025[["Driver_AvgFinish_Last5","Driver_PodiumRate_Last5","Driver_PrevFinish"]]
latest_2025["Podium_Prob"] = model.predict_proba(X_pred)[:, 1]

top3 = (latest_2025.sort_values("Podium_Prob", ascending=False)
        .head(3)[["Driver Name","Team","Podium_Prob"]]
        .rename(columns={"Driver Name":"Driver","Podium_Prob":"Prob"}))

print("\n=== Predicted Top 3 (next race form, from latest 2025 rows) ===")
for i, row in enumerate(top3.itertuples(index=False), start=1):
    print(f"{i}. {row.Driver} ({row.Team}) — podium prob: {row.Prob:.3f}")

print(f"\nMAE (Test: 2025 Italy + Netherlands): {mae_test:.4f}")
