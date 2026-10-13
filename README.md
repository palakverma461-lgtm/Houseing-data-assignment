# House Price Prediction Using Linear Regression

## 1. Project Overview
This project uses Machine Learning to predict house prices from `Housing.csv`. Each row represents a house. The model learns from house features and predicts the numerical value in the `price` column.

**Problem type:** Regression, because the target is a numerical price.

## 2. Technologies Used
- **Python:** programming language.
- **Pandas:** loads and handles tabular data.
- **Scikit-learn:** splits data, trains the model, and provides evaluation tools.
- **Linear Regression:** example algorithm for predicting numerical house prices.

Install libraries if needed:
```bash
pip install pandas scikit-learn
```

## 3. Dataset
The dataset is `Housing.csv`.
- **Input features (`X`):** all columns except `price`.
- **Target (`y`):** the `price` column.
The exact input features depend on the columns in your dataset.

## 4. Code Explanation

### A. Separate input and output
```python
X = df.drop("price", axis=1)
y = df["price"]
```
- `X` contains all input columns except `price`.
- `df.drop("price", axis=1)` removes the `price` column.
- `axis=1` means a column operation (`axis=0` refers to rows).
- `y` stores the actual house prices that the model will learn to predict.

**Why?** The model should learn from house features and predict the price. The answer (`price`) must not also be included in the inputs.

### B. Import the split function
```python
from sklearn.model_selection import train_test_split
```
This imports a Scikit-learn function that divides data into training and testing sets.

**Why?** The model is trained on one portion and evaluated on separate data it has not trained on.

### C. Split the dataset
```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```
- `X_train`: features used for training.
- `X_test`: features reserved for testing.
- `y_train`: correct prices for training rows.
- `y_test`: actual prices for test rows.
- `test_size=0.20`: 20% of rows are kept for testing; 80% are used for training.
- `random_state=42`: makes the random split reproducible when the dataset and environment are unchanged. `42` is simply a chosen seed.

**Why an 80/20 split?** Most of the data is available for learning while some remains unseen for evaluation.

### D. Display split sizes
```python
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
```
`.shape` shows `(number_of_rows, number_of_columns)`. It helps check the size of the training and testing feature sets. Exact values depend on your dataset.

### E. Train the model
```python
model.fit(X_train, y_train)
```
- `model` is the estimator created earlier, for example `LinearRegression()`.
- `.fit()` trains it using input features and their correct prices.
- Linear Regression estimates coefficients to model the relationship between features and price.

**Why?** This is the learning step.

For example, define the model earlier:
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

**Important:** If the dataset contains text/category columns, convert them to numeric features first (for example, with `pd.get_dummies()` or a preprocessing pipeline). This snippet assumes the data is already prepared.

### F. Predict test prices
```python
y_pred = model.predict(X_test)
```
- `.predict()` uses the trained model to estimate prices for the test rows.
- `y_pred` contains predicted prices.
- `y_test` contains the actual prices for those same rows.

**Why?** Comparing `y_pred` with `y_test` shows how well the model predicts unseen data.

## 5. Workflow
1. Load `Housing.csv` into a Pandas DataFrame named `df`.
2. Clean data and encode text/category columns as needed.
3. Separate input features (`X`) and target (`y`).
4. Split into 80% training and 20% testing.
5. Train using `model.fit()`.
6. Predict using `model.predict()`.
7. Evaluate predictions with regression metrics.

## 6. Training vs Testing

| Term | Meaning |
|---|---|
| Training data | Data the model learns from. |
| Testing data | Unseen data used to evaluate the trained model. |
| `y_test` | Actual prices in the test set. |
| `y_pred` | Prices predicted by the model. |

## 7. Optional: Evaluate the Model
House-price prediction is regression, so use regression metrics rather than a classification report or confusion matrix for raw prices.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
```

- **MAE:** average absolute prediction error.
- **MSE:** average squared prediction error; larger errors weigh more.
- **RMSE:** square root of MSE, in the same units as price.
- **R² score:** compares the model with a mean-prediction baseline; it can be negative if predictions are poor.

These metrics are optional additions; they are not part of the snippet you shared.

## 8. Common Errors
- `df is not defined`: load the CSV into `df` first.
- `model is not defined`: create the model before `.fit()`.
- Error about strings: encode categorical columns as numeric features before fitting.
- Feature mismatch: make sure train and test data have identical columns, order, and preprocessing.
- A **confusion matrix/classification report** is not appropriate for continuous house prices unless you first turn prices into categories such as High/Low.

## 9. Short Viva Answer
“This project predicts house prices using Linear Regression. I separated the input features (`X`) from the target (`y`). Then I used `train_test_split` to divide the data into 80% training and 20% testing. `random_state=42` makes the split reproducible. I trained the model using `fit()` and predicted prices for the test set using `predict()`. I can then evaluate predictions with MAE, MSE, RMSE, and R².”

## 10. Suggested Folder Structure
```text
Housing-Project/
├── Housing.csv
├── main.py
└── README.md
```
Make sure `Housing.csv` is available at the path used by your Python code, and define `df` and `model` earlier in `main.py`.
