# 📊 Regression Analyzer & Visualizer

This project is a simple but functional linear regression tool built with **Python** and **Streamlit**. It supports both a web interface and a command-line interface for uploading datasets, training a regression model using gradient descent, and visualizing the results.

---

## 💡 What It Does

* Lets you upload your own `.csv` dataset
* Allows you to select feature(s) (predictors) and a target variable
* Performs **Simple or Multiple Linear Regression**
* Uses **gradient descent** (implemented from scratch)
* Plots real vs predicted values
* Works both via **web interface (Streamlit)** and **command line (`main.py`)**

---

## 📦 Project Structure

```
regression-project/
├── stream.py           # Streamlit app (UI)
├── main.py             # CLI interface to run regression manually
├── regressor.py        # Core regression logic (gradient descent)
├── error.py            # Evaluation metrics (MSE, R2)
├── reader.py           # CSV reader for loading selected columns
├── plotter.py          # Visualization functions
├── learning_rater.py   # Learning rate utility (optional)
├── requirements.txt    # List of required Python packages
└── README.md           # This file
```

---

## ✅ Requirements

Install required packages using:

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**

```text
streamlit>=1.24.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
```

---

## 🚀 How to Use

### ▶️ Option 1: Run via Streamlit UI

```bash
streamlit run stream.py
```

**Features:**

* Upload your `.csv` file
* Select predictors and target column
* Click **"Analyze"** to train the model
* View results in an interactive regression plot

---

### 💻 Option 2: Run via Command Line (`main.py`)

This is a non-interactive version. You manually specify the file path and columns to use.

**Example `main.py` usage:**

```python
from matplotlib import pyplot as plt
from regressor import Regressor
from reader import Reader
from error import Error

# Define your columns
predictors = ["predictor_1"]
targets = ["target_1"]

# Read selected columns from CSV
X, y = Reader.partial_reading("FilePath", predictors, targets)

# Train model
model = Regressor()
beta = model.gradient_descent(X, y, learning_rate=0.000006)
print(beta)

# Make predictions
y_pred = model.predict(X, beta)

# Evaluate
mse = Error.mean_squared_error(y, y_pred)
r2 = Error.r2_score(y, y_pred)
print("Mean Squared Error (MSE):", mse)
print("R2 score:", r2)

# Plot results
plt.scatter(y, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel(", ".join(predictors))
plt.ylabel(", ".join(targets))
plt.title("Model Performance")

plt.text(0.05, 0.95, f"MSE = {mse:.2f}", transform=plt.gca().transAxes,
         fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
plt.text(0.05, 0.85, f"R2 = {r2:.2f}", transform=plt.gca().transAxes,
         fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
plt.show()
```

**📝 Note:** Replace `"FilePath"` with your actual `.csv` file path, and use the correct column names for `"predictor_1"` and `"target_1"`.

---

## 📌 Streamlit App Instructions

* Click "Browse files" to upload your CSV dataset
* A preview of your data will be shown automatically
* Select column(s) as predictor(s)
* Select one column as the target
* Click **"Analyze"** to train the model
* View the result plot:

  * 🔵 Blue dots = actual values
  * 🔴 Red line = regression line

---

## 🧠 What You’ll Learn

* How linear regression works under the hood
* How to build a regression model from scratch using NumPy
* How to evaluate model performance using MSE and R²
* How to build a clean, interactive ML interface with Streamlit

---

## ⚠️ Limitations

* ❌ This project does not use scikit-learn — built fully from scratch
* 🧹 Data must be clean and numeric (NaNs are dropped automatically)
* ➕ Only supports linear regression (no polynomial or logistic regression yet)

---

## 🔮 Future Improvements

* Support multiple target variables
* Add feature normalization toggle to UI
* Export trained model to file
* Add polynomial regression support
* Output predictions as downloadable CSV

---

## 🗃️ License

This project is licensed under the MIT License.
Feel free to use, modify, and share it however you'd like.
