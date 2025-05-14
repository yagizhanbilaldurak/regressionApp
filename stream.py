import streamlit as st
import pandas as pd

from regressor import Regressor
from plotter import Plotter

st.title("Regression Analyzer & Plotter")

col1, col2 = st.columns([1, 2])

# Left side: Upload and train
with col1:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Success!")
        st.dataframe(df.head())

        columns = df.columns.tolist()

        predictors = st.multiselect("Choose predictor(s)", columns, default=columns)
        target = st.selectbox("Choose target", columns)

        if st.button("Analyze"):
            if len(predictors) == 0:
                st.error("Pick at least one predictor.")
            else:
                X = df[predictors].values
                y = df[[target]].values

                model = Regressor()
                beta = model.gradient_descent(X, y, learning_rate=0.001, iterations=1000)

                st.session_state["X"] = X
                st.session_state["y"] = y
                st.session_state["beta"] = beta
                st.session_state["predictors"] = predictors
                st.session_state["target"] = target

                st.success("Training complete!")

# Right side: Plot
with col2:
    st.header("Regression Plot")

    if "beta" in st.session_state:
        X = st.session_state["X"]
        y = st.session_state["y"]
        beta = st.session_state["beta"]
        predictors = st.session_state["predictors"]
        target = st.session_state["target"]

        plotter = Plotter()

        feature_index = 0
        target_index = 0
        feature_name = predictors[feature_index]
        target_name = target

        fig = plotter.plot_streamlit(
            X, y, beta,
            feature_index=feature_index,
            target_index=target_index,
            feature_name=feature_name,
            target_name=target_name
        )
        st.pyplot(fig)
