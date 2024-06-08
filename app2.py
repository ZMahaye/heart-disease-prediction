import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model and preprocessor
model = joblib.load('best_model1.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Define the prediction function
def predict_heart_disease(features):
    # Convert the features list to a DataFrame
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    features_df = pd.DataFrame([features], columns=columns)
    
    # Apply the preprocessing pipeline to the input features
    features_preprocessed = preprocessor.transform(features_df)
    
    # Make prediction
    return model.predict(features_preprocessed)[0]

# Streamlit application
st.title("Heart Disease Prediction")

st.sidebar.header("Patient Details")

# Input fields for patient details
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia", options=[0, 1, 2, 3])

# Predict button
if st.sidebar.button("Predict"):
    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    prediction = predict_heart_disease(features)
    if prediction == 1:
        st.write("The patient is likely to have heart disease.")
    else:
        st.write("The patient is unlikely to have heart disease.")

# Error handling for unexpected inputs
try:
    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    prediction = predict_heart_disease(features)
except Exception as e:
    st.error(f"An error occurred: {e}")
