import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "insurance_model.joblib"

st.set_page_config(page_title="Insurance Premium Predictor", page_icon="💳")
st.title("💳 Insurance Premium Predictor")
st.write("Enter customer details to estimate the insurance premium.")

if not MODEL_PATH.exists():
    st.warning("Model not found. Run `python src/model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["female", "male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict Premium"):
    input_df = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker, "region": region
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated annual premium: ${prediction:,.2f}")
