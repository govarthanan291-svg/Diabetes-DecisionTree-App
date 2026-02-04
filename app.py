import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #74ebd5, #ACB6E5);
}

.title {
    text-align: center;
    color: #2C3E50;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #34495E;
    font-size: 18px;
}

.predict-btn button {
    background-color: #27AE60;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>🩺 Diabetes Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Decision Tree Based ML Application</div><br>", unsafe_allow_html=True)

# Load model
model = joblib.load("diabetes_model.pkl")

# Inputs
preg = st.slider("Pregnancies", 0, 20)
glu = st.slider("Glucose Level", 0, 200)
bp = st.slider("Blood Pressure", 0, 150)
skin = st.slider("Skin Thickness", 0, 100)
ins = st.slider("Insulin", 0, 900)
bmi = st.slider("BMI", 0.0, 70.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0)
age = st.slider("Age", 1, 120)

# Predict
st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
if st.button("Predict"):
    data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    result = model.predict(data)

    if result[0] == 1:
        st.error("⚠️ High chance of Diabetes")
    else:
        st.success("✅ No Diabetes Detected")
st.markdown("</div>", unsafe_allow_html=True)
