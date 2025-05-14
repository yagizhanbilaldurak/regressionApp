# 📊 Regression Analyzer & Visualizer

This project is a simple but functional linear regression tool built with **Python** and **Streamlit**. It supports both a web interface and a command-line interface for uploading datasets, training a regression model using gradient descent, and visualizing the results.

---

## 💡 What It Does

- Lets you upload your own `.csv` dataset
- Allows you to select feature(s) (predictors) and a target variable
- Performs **Simple or Multiple Linear Regression**
- Uses **gradient descent** (implemented from scratch)
- Plots real vs predicted values
- Works both via **web interface (Streamlit)** and **command line (`main.py`)**

---

## 📦 Project Structure

regression-project/
├── stream.py # Streamlit app (UI)
├── main.py # CLI interface to run regression manually
├── regressor.py # Core regression logic (gradient descent)
├── error.py # Evaluation metrics (MSE, R2)
├── reader.py # CSV reader for loading selected columns
├── plotter.py # Visualization functions
├── learning_rater.py # Learning rate utility (optional)
├── requirements.txt # List of required Python packages
└── README.md # This file

---

## ✅ Requirements

Install required packages using:

```bash
pip install -r requirements.txt

content of requirements.txt:
streamlit>=1.24.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0


