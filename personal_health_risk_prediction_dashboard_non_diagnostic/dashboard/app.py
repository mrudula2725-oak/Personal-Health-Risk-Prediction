import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Hypertension Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# Risk mapping
risk_map = {
    0: "Low",
    1: "Moderate",
    2: "High"
}

# Title
st.title("❤️ Personal Health Risk Prediction Dashboard")
st.markdown("This tool predicts **Hypertension Risk Level** based on patient health indicators. *(Non-Diagnostic Tool)*")

# Load model

model = joblib.load("../models/Personal_Risk_ML.pkl")
scaler = joblib.load("../models/scaler.pkl")

# Sidebar
st.sidebar.header("Patient Health Information")

gender = st.sidebar.selectbox("Gender",["Male","Female"])
age = st.sidebar.slider("Age",20,90,45)

systolic = st.sidebar.slider("Systolic Blood Pressure",90,200,120)
diastolic = st.sidebar.slider("Diastolic Blood Pressure",60,120,80)

salt = st.sidebar.selectbox("Salt Intake",["Low","Moderate","High"])
alcohol = st.sidebar.selectbox("Alcohol Consumption",["No","Yes"])

stress = st.sidebar.slider("Stress Level",1,10,5)

# Encoding
gender = 1 if gender=="Male" else 0
salt_map = {"Low":0,"Moderate":1,"High":2}
salt = salt_map[salt]
alcohol = 1 if alcohol=="Yes" else 0

# Create dataframe
input_data = pd.DataFrame([[gender,age,systolic,diastolic,salt,alcohol,stress]],
columns=[
"Gender",
"Age",
"Systolic_BP",
"Diastolic_BP",
"Salt_Intake",
"Alcohol_Consumption",
"Stress_Level"
])

# Scale
scaled = scaler.transform(input_data)

# Prediction section
st.subheader("Prediction")

if st.button("Predict Risk Level"):

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]

    risk_label = risk_map[prediction]

    # Risk display
    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric("Predicted Risk", risk_label)

    with col2:
        st.metric("Systolic BP", systolic)

    with col3:
        st.metric("Stress Level", stress)

    # Risk message
    if risk_label == "Low":
        st.success("Low Hypertension Risk")
    elif risk_label == "Moderate":
        st.warning("Moderate Hypertension Risk")
    else:
        st.error("High Hypertension Risk")

    # Probability charts
    prob_df = pd.DataFrame({
        "Risk Level":["Low","Moderate","High"],
        "Probability":probability
    })

    col1,col2 = st.columns(2)

    with col1:
        fig = px.bar(
            prob_df,
            x="Risk Level",
            y="Probability",
            title="Risk Probability Distribution"
        )
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        fig2 = px.pie(
            prob_df,
            values="Probability",
            names="Risk Level",
            title="Risk Distribution"
        )
        st.plotly_chart(fig2,use_container_width=True)

# Dataset insights
st.header("Dataset Insights")

data = pd.read_csv("../data/hypertension_data.csv")

col1,col2 = st.columns(2)

# Age vs BP
with col1:
    fig3 = px.scatter(
        data,
        x="Age",
        y="Systolic_BP",
        color="Risk_Level",
        title="Age vs Systolic Blood Pressure"
    )
    st.plotly_chart(fig3,use_container_width=True)

# Stress vs Risk
with col2:
    fig4 = px.box(
        data,
        x="Risk_Level",
        y="Stress_Level",
        title="Stress Level vs Risk"
    )
    st.plotly_chart(fig4,use_container_width=True)

# BP distribution
fig5 = px.histogram(
    data,
    x="Systolic_BP",
    color="Risk_Level",
    title="Systolic Blood Pressure Distribution"
)

st.plotly_chart(fig5,use_container_width=True)

# Footer
st.markdown("---")
st.markdown("ML Models Used: Logistic Regression | Random Forest | SVM")
st.markdown("Dataset: Hypertension Health Indicators")