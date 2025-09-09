import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, mean_absolute_error,
    brier_score_loss, accuracy_score
)
import matplotlib.pyplot as plt
import numpy as np

# load engineered data
df = pd.read_csv("data/F1_RaceResult_DriverFeatures.csv")

# drop early rows without features
df = df.dropna(subset=["Driver_AvgFinish_Last5","Driver_PodiumRate_Last5","Driver_PrevFinish"])

# features & target
X = df[["Driver_AvgFinish_Last5","Driver_PodiumRate_Last5","Driver_PrevFinish"]]
y = df["Podium"]

# train/test split: all before 2025 → train, 2025 races → test
X_train = X[df["Year"] < 2025]
y_train = y[df["Year"] < 2025]
X_test  = X[df["Year"] == 2025]
y_test  = y[df["Year"] == 2025]

# define model
model = XGBClassifier(
    n_estimators=125,
    learning_rate=0.02,
    max_depth=3,
    subsample=0.6,
    colsample_bytree=0.5,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
)

# train
model.fit(X_train, y_train)

# ===== Train evaluation =====
y_pred_train = model.predict(X_train)
y_prob_train = model.predict_proba(X_train)[:,1]

print("=== TRAIN RESULTS (<=2024) ===")
print(classification_report(y_train, y_pred_train))
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Train ROC-AUC:", roc_auc_score(y_train, y_prob_train))
print("Train MAE:", mean_absolute_error(y_train, y_prob_train))
print("Train Brier Score:", brier_score_loss(y_train, y_prob_train))

# ===== Test evaluation =====
y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)[:,1]

print("\n=== TEST RESULTS (2025) ===")
print(classification_report(y_test, y_pred_test))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob_test))
print("Test MAE:", mean_absolute_error(y_test, y_prob_test))
print("Test Brier Score:", brier_score_loss(y_test, y_prob_test))

# ===== Feature importances =====
importance = model.feature_importances_
features = X_train.columns

plt.figure(figsize=(6,4))
plt.barh(features, importance)
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()
