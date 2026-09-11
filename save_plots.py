import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# Paths
PROJECT_DIR = Path(r"C:\Users\djato\smart-irrigation-ml")
IMAGES_DIR = PROJECT_DIR / "images"
DATA_PATH = Path(r"C:\Users\djato\Downloads\peerj-cs-10-2112-s002.csv")

IMAGES_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH, encoding="gb18030")

# Operational target
df["Recommended_Irrigation_mm"] = df["Amount of irrigation"].clip(lower=0)

# Date features
df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day_of_year"] = df["Date"].dt.dayofyear

# Load saved feature list and model
features = joblib.load(PROJECT_DIR / "smart_irrigation_features.joblib")
model = joblib.load(PROJECT_DIR / "smart_irrigation_random_forest.joblib")

X = df[features]
y = df["Recommended_Irrigation_mm"]

# Same temporal test period
test_mask = df["Year"] >= 2021

X_test = X[test_mask]
y_test = y[test_mask]

# Predictions
y_pred = model.predict(X_test)

# -------------------------
# Actual vs Predicted
# -------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Irrigation (mm)")
plt.ylabel("Predicted Irrigation (mm)")
plt.title("Random Forest: Actual vs Predicted")
plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -------------------------
# Feature Importance
# -------------------------
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(9, 6))

plt.barh(
    importance["Feature"][::-1],
    importance["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("DONE")
print(IMAGES_DIR / "actual_vs_predicted.png")
print(IMAGES_DIR / "feature_importance.png")