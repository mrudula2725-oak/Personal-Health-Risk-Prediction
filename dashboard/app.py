import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# ---------- Page config ----------
st.set_page_config(
    page_title="Hypertension Risk Prediction",
    page_icon="❤️",
    layout="wide",
)

# ---------- Paths (work locally and on Streamlit Cloud) ----------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
MODEL_PATH = BASE_DIR / "models" / "Personal_Risk_ML.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
DATA_PATH = BASE_DIR / "data" / "hypertension_data.csv"

# ---------- Risk mapping ----------
risk_map = {0: "Low", 1: "Moderate", 2: "High"}
risk_order = ["Low", "Moderate", "High"]


# ---------- Cached loaders ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Show text labels in charts if Risk_Level is stored as 0/1/2
    if pd.api.types.is_numeric_dtype(df["Risk_Level"]):
        df["Risk_Level"] = df["Risk_Level"].map(risk_map)
    return df


try:
    model, scaler = load_artifacts()
    data = load_data()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

# ---------- Title ----------
st.title("❤️ Personal Health Risk Prediction Dashboard")
st.markdown(
    "This tool predicts **Hypertension Risk Level** from health indicators. "
    "*(Non-Diagnostic Tool)*"
)
st.warning(
    "Educational use only. This tool is not medical advice or a diagnosis. "
    "Consult a doctor for any health concerns."
)

# ---------- Sidebar inputs ----------
st.sidebar.header("Your Health Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 20, 90, 45)
systolic = st.sidebar.slider("Systolic Blood Pressure", 90, 200, 120)
diastolic = st.sidebar.slider("Diastolic Blood Pressure", 60, 120, 80)
salt = st.sidebar.selectbox("Salt Intake", ["Low", "Moderate", "High"])
alcohol = st.sidebar.selectbox("Alcohol Consumption", ["No", "Yes"])
stress = st.sidebar.slider("Stress Level", 1, 10, 5)

# ---------- Encoding ----------
gender_val = 1 if gender == "Male" else 0
salt_val = {"Low": 0, "Moderate": 1, "High": 2}[salt]
alcohol_val = 1 if alcohol == "Yes" else 0

input_data = pd.DataFrame(
    [[gender_val, age, systolic, diastolic, salt_val, alcohol_val, stress]],
    columns=[
        "Gender",
        "Age",
        "Systolic_BP",
        "Diastolic_BP",
        "Salt_Intake",
        "Alcohol_Consumption",
        "Stress_Level",
    ],
)

scaled = scaler.transform(input_data)

# ---------- Prediction ----------
st.subheader("Prediction")

if st.button("Predict Risk Level"):
    prediction = int(model.predict(scaled)[0])
    risk_label = risk_map[prediction]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Risk", risk_label)
    with col2:
        st.metric("Systolic BP", systolic)
    with col3:
        st.metric("Stress Level", stress)

    if risk_label == "Low":
        st.success("Low Hypertension Risk")
    elif risk_label == "Moderate":
        st.warning("Moderate Hypertension Risk")
    else:
        st.error("High Hypertension Risk")

    # Probability charts (needs a model that supports predict_proba)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled)[0]
        prob_df = pd.DataFrame(
            {"Risk Level": risk_order, "Probability": probability}
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                prob_df,
                x="Risk Level",
                y="Probability",
                title="Risk Probability Distribution",
            )
            st.plotly_chart(fig)
        with c2:
            fig2 = px.pie(
                prob_df,
                values="Probability",
                names="Risk Level",
                title="Risk Distribution",
            )
            st.plotly_chart(fig2)
    else:
        st.info("This model does not provide probability scores.")

# ---------- Dataset insights ----------
st.header("Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    fig3 = px.scatter(
        data,
        x="Age",
        y="Systolic_BP",
        color="Risk_Level",
        category_orders={"Risk_Level": risk_order},
        title="Age vs Systolic Blood Pressure",
    )
    st.plotly_chart(fig3)

with col2:
    fig4 = px.box(
        data,
        x="Risk_Level",
        y="Stress_Level",
        category_orders={"Risk_Level": risk_order},
        title="Stress Level vs Risk",
    )
    st.plotly_chart(fig4)

fig5 = px.histogram(
    data,
    x="Systolic_BP",
    color="Risk_Level",
    category_orders={"Risk_Level": risk_order},
    title="Systolic Blood Pressure Distribution",
)
st.plotly_chart(fig5)

# ---------- Footer ----------
st.markdown("---")
st.markdown("ML Models Used: Logistic Regression | Random Forest | SVM")
st.markdown("Dataset: Hypertension Health Indicators")