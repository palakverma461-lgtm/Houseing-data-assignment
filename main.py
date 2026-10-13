
# ==========================================
# HOUSE PRICE PREDICTION USING LINEAR REGRESSION
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
    accuracy_score
)

# 1. LOAD DATASET
df = pd.read_csv("Housing.csv")

print("\n========== ORIGINAL DATA ==========")
print(df.head())
print("Dataset Shape:", df.shape)

# 2. DATA CLEANING
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# Convert target price to numeric
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Remove rows with missing target price
df = df.dropna(subset=["price"])

# Handle missing numeric values
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(
    df[numeric_cols].median()
)

# Handle missing categorical values
categorical_cols = df.select_dtypes(
    exclude=np.number
).columns

for col in categorical_cols:
    if df[col].isnull().any():
        mode_values = df[col].mode()
        if not mode_values.empty:
            df[col] = df[col].fillna(mode_values.iloc[0])

print("\n========== CLEANED DATA ==========")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nCleaned Dataset Shape:", df.shape)

# 3. CONVERT CATEGORICAL DATA TO NUMERIC
df = pd.get_dummies(df, drop_first=True, dtype=int)

# 4. SEPARATE FEATURES AND TARGET
X = df.drop(columns=["price"])
y = df["price"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# 5. SPLIT DATA INTO TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# 6. TRAIN LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")

# 7. PREDICT HOUSE PRICES
y_pred = model.predict(X_test)

# 8. REGRESSION EVALUATION
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== REGRESSION RESULTS ==========")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# 9. ACTUAL VS PREDICTED HOUSE PRICES
results = pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": y_pred.round(2),
    "Difference": (y_test.to_numpy() - y_pred).round(2)
})

print("\n========== ACTUAL VS PREDICTED ==========")
print(results.head(10).to_string(index=False))

results.to_csv("housing_predictions.csv", index=False)
print("\nPredictions saved to housing_predictions.csv")

# 10. CONFUSION MATRIX AND CLASSIFICATION REPORT
# These require categories, not continuous prices.
# Use the training-set median as the High/Low threshold.

threshold = y_train.median()

actual_class = np.where(
    y_test.to_numpy() >= threshold, 1, 0
)

predicted_class = np.where(
    y_pred >= threshold, 1, 0
)

print("\n========== CLASSIFICATION RESULTS ==========")
print(f"High/Low Price Threshold: {threshold:.2f}")

print("\nClassification Report:")
print(classification_report(
    actual_class,
    predicted_class,
    labels=[0, 1],
    target_names=["Low Price", "High Price"],
    zero_division=0
))

print("Classification Accuracy:",
      round(accuracy_score(actual_class, predicted_class), 4))

cm = confusion_matrix(
    actual_class,
    predicted_class,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(pd.DataFrame(
    cm,
    index=["Actual Low", "Actual High"],
    columns=["Predicted Low", "Predicted High"]
))

print("\n========== PROGRAM COMPLETED ==========")
