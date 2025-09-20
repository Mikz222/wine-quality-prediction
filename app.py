import streamlit as st
import pandas as pd
import joblib

# Load trained model (make sure you saved it in notebook.ipynb)
model = joblib.load("artifacts/model.pkl")

st.title("🍷 Wine Quality Prediction")

st.write("Enter the wine’s chemical measurements below:")

# Example input fields
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.number_input("Citric Acid", 0.0, 2.0, 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.0, 15.0, 1.9)
chlorides = st.number_input("Chlorides", 0.0, 0.2, 0.076)
alcohol = st.number_input("Alcohol", 8.0, 15.0, 9.4)

if st.button("Predict Quality"):
    features = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, alcohol]]
    prediction = model.predict(features)
    st.success(f"Predicted Wine Quality: {prediction[0]}")
