import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the web app
st.title('Bankruptcy Prediction App')

# Input features
industrial_risk = st.selectbox('Industrial Risk', [0, 0.5, 1])
management_risk = st.selectbox('Management Risk', [0, 0.5, 1])
financial_flexibility = st.selectbox('Financial Flexibility', [0, 0.5, 1])
credibility = st.selectbox('Credibility', [0, 0.5, 1])
competitiveness = st.selectbox('Competitiveness', [0, 0.5, 1])
operating_risk = st.selectbox('Operating Risk', [0, 0.5, 1])

# Prediction button
if st.button('Predict'):
    # Create a feature array
    features = np.array([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]])

    # Make prediction
    prediction = model.predict(features)

    # Display prediction
    st.write('Prediction:', prediction)
