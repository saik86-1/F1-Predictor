import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# load engineered dataset
df = pd.read_csv("data/F1_RaceResult_DriverFeatures.csv")

# drop rows with missing engineered features (early races with <5 history)
df = df.dropna(subset=["Driver_AvgFinish_Last5","Driver_PodiumRate_Last5","Driver_PrevFinish"])

# features and target
X = df[["Driver_AvgFinish_Last5","Driver_PodiumRate_Last5","Driver_PrevFinish"]]
y = df["Podium"]

# time-based split: train = before 2025, test = 2025 races
X_train = X[df["Year"] < 2025]
y_train = y[df["Year"] < 2025]
X_test  = X[df["Year"] == 2025]
y_test  = y[df["Year"] == 2025]

# model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("Classification report (2025 test set):")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

import matplotlib.pyplot as plt

# feature importances
importances = model.feature_importances_
features = X_train.columns

print("\nFeature importances:")
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.4f}")

# plot
plt.figure(figsize=(6,4))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance (Gradient Boosting)")
plt.tight_layout()
plt.show()
