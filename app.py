import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

# --- 1. Optimization: Caching Model Loading ---
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

model_path = 'diabetes_xgb_model_v1.joblib'
model = load_model(model_path)

# --- 2. Input Interface ---
def get_user_inputs():
    st.sidebar.header("📋 Patient Information")
    
    with st.sidebar.expander("🩸 Blood Test Results", expanded=True):
        chol = st.number_input("Total Cholesterol (mg/dL)", 100, 500, 200)
        hdl = st.number_input("HDL - Good Cholesterol (mg/dL)", 10, 120, 50)
        stab_glu = st.number_input("Stabilized Glucose (mg/dL)", 40, 400, 100)
        
    with st.sidebar.expander("📏 Physical Measurements", expanded=True):
        age = st.slider("Age", 18, 100, 45)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
        waist = st.slider("Waist (cm)", 20, 150, 80)
        hip = st.slider("Hip (cm)", 20, 150, 95)

    # Feature Engineering 
    ratio = stab_glu / hdl
    bmi = weight / ((height/100) ** 2)
    
    data = {
        'chol': chol, 'stab.glu': stab_glu, 'hdl': hdl, 'ratio': ratio,
        'age': age, 'waist': waist, 'hip': hip, 'bmi': bmi
    }
    return pd.DataFrame([data])

# --- 3. MAIN DASHBOARD ---
st.title("🩺 Diabetes Risk Prediction")
st.markdown(f"**Developer:** Luke Vu | **Model:** XGBoost Regressor")
st.divider()

if model is None:
    st.error(f"❌ Model file not found at `{model_path}`. Please check your directory.")
    st.stop()

input_df = get_user_inputs()

# Show key metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Calculated BMI", f"{input_df['bmi'].iloc[0]:.2f}")
with col2:
    st.metric("Glucose/HDL Ratio", f"{input_df['ratio'].iloc[0]:.2f}")
with col3:
    st.metric("Age Group", f"{input_df['age'].iloc[0]} yrs")

st.subheader("📋 Input Summary")
st.dataframe(input_df, use_container_width=True)

# --- 4. PREDICTION LOGIC ---
if st.button("Analyze Risk Level", type="primary", use_container_width=True):
    prediction = model.predict(input_df)
    glyhb = prediction[0]
    
    st.markdown("---")
    
    # Results Display
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.subheader("Result:")
        if glyhb >= 6.5:
            st.error(f"### {glyhb:.2f}% (Diabetes)")
        elif glyhb >= 5.7:
            st.warning(f"### {glyhb:.2f}% (Pre-diabetes)")
        else:
            st.success(f"### {glyhb:.2f}% (Healthy)")
            

# --- 5. FOOTER ---
st.caption("Disclaimer: This tool is for educational purposes only and not a substitute for professional medical advice.")