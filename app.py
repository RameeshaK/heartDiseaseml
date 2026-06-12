import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Cardiovascular Risk Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Cardiovascular Disease Risk Predictor")
st.write("Enter patient clinical metrics below to estimate the risk of heart disease.")

@st.cache_data
def load_and_train_model():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
    df = pd.read_csv(url)
    
    X = df.drop(columns=['target'])
    y = df['target']
    X = pd.get_dummies(X, drop_first=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns

model, scaler, feature_columns = load_and_train_model()

st.header("Patient Clinical Metrics")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 50)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cp = st.slider("Chest Pain Type (0-4)", 0, 4, 1)
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)

with col2:
    chol = st.slider("Serum Cholestoral (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["False", "True"])
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])

input_data = {
    'age': age,
    'sex': 1 if sex == "Male" else 0,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': 1 if fbs == "True" else 0,
    'restecg': 0, # Default filler values for complex columns not in basic UI
    'thalach': thalach,
    'exang': 1 if exang == "Yes" else 0,
    'oldpeak': 0.0,
    'slope': 1,
    'ca': 0,
    'thal': 2
}

input_df = pd.DataFrame([input_data])

for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]

if st.button("Calculate Cardiovascular Risk", type="primary"):
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **High Risk Detected.** The model estimates a **{probability*100:.1f}%** probability of cardiovascular disease. Clinical follow-up recommended.")
    else:
        st.success(f"✅ **Low Risk Detected.** The model estimates a **{probability*100:.1f}%** probability of cardiovascular disease. Maintain a healthy lifestyle!")
