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

# --- 2. Helper Functions for Conversion ---
# Tại sao cần Helper functions? Để code sạch sẽ và có thể tái sử dụng.
def convert_units():
    with st.sidebar.expander("🔄 Quick Unit Converter", expanded=False):
        st.write("Convert your local units to model units:")
        
        # 1. Weight: lbs to kg
        lbs = st.number_input("Weight (lbs)", value=154.0)
        kg_res = lbs * 0.453592
        st.info(f"➡️ **{kg_res:.2f} kg**")
        
        st.divider()
        
        # 2. Length: inches to cm
        inches = st.number_input("Length (inch)", value=67.0)
        cm_res = inches * 2.54
        st.info(f"➡️ **{cm_res:.2f} cm**")
        
        st.divider()
        
        # 3. Glucose/Chol: mmol/L to mg/dL
        # Note: Glucose factor is 18.0, Cholesterol is 38.67
        mmol = st.number_input("Value (mmol/L)", value=5.5)
        st.write("Convert to:")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            glu_mg = mmol * 18.018
            st.caption("Glucose")
            st.code(f"{glu_mg:.1f} mg/dL")
        with col_c2:
            chol_mg = mmol * 38.67
            st.caption("Cholesterol")
            st.code(f"{chol_mg:.1f} mg/dL")

# --- 3. Input Interface ---
def get_user_inputs():
    st.sidebar.header("📋 Patient Information")
    
    # Tích hợp bộ chuyển đổi vào Sidebar
    convert_units()
    
    with st.sidebar.expander("🩸 Blood Test Results", expanded=True):
        chol = st.number_input("Total Cholesterol (mg/dL)", 100, 500, 200)
        hdl = st.number_input("HDL (mg/dL)", 10, 120, 50)
        stab_glu = st.number_input("Stabilized Glucose (mg/dL)", 40, 400, 100)
        
    with st.sidebar.expander("📏 Physical Measurements", expanded=True):
        age = st.slider("Age", 18, 100, 45)
        waist = st.slider("Waist (cm)", 20, 150, 80)
        hip = st.slider("Hip (cm)", 20, 150, 95)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)

    # Feature Engineering 
    ratio = stab_glu / hdl
    bmi = weight / ((height/100) ** 2)
    
    data = {
        'chol': chol, 'stab.glu': stab_glu, 'hdl': hdl, 'ratio': ratio,
        'age': age, 'waist': waist, 'hip': hip, 'bmi': bmi
    }
    return pd.DataFrame([data])

# --- 4. MAIN DASHBOARD ---
st.title("🩺 Diabetes Risk Prediction")
st.markdown(f"**Developer:** Luke Vu | **Model:** XGBoost Regressor")

# Adding a reference conversion table in the main area
with st.expander("📚 Reference Unit Conversion Table"):
    st.markdown("""
    | Measurement | From Unit | Formula | To Unit (Model) |
    | :--- | :--- | :--- | :--- |
    | **Glucose** | 1 mmol/L | x 18.0 | mg/dL |
    | **Cholesterol** | 1 mmol/L | x 38.67 | mg/dL |
    | **Weight** | 1 lb (pound) | / 2.205 | kg |
    | **Length** | 1 inch | x 2.54 | cm |
    """)

st.divider()

if model is None:
    st.error(f"❌ Model file not found at `{model_path}`. Please check your directory.")
    st.stop()

input_df = get_user_inputs()

# Metrics Display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Calculated BMI", f"{input_df['bmi'].iloc[0]:.2f}")
with col2:
    st.metric("Glucose/HDL Ratio", f"{input_df['ratio'].iloc[0]:.2f}")
with col3:
    st.metric("Age Group", f"{input_df['age'].iloc[0]} yrs")

st.subheader("📋 Input Summary")
st.dataframe(input_df, use_container_width=True)

# --- 5. PREDICTION LOGIC ---
if st.button("Analyze Risk Level", type="primary", use_container_width=True):
    prediction = model.predict(input_df)
    glyhb = prediction[0]
    
    st.markdown("---")
    
    col_res1, col_res2 = st.columns([1, 2])
    with col_res1:
        st.subheader("Predicted result of percentage Glycated Hemoglobin - average blood sugar levels over 2-3 months:")
        if glyhb >= 6.5:
            st.error(f"### {glyhb:.2f}% (Diabetes)")
        elif glyhb >= 5.7:
            st.warning(f"### {glyhb:.2f}% (Pre-diabetes)")
        else:
            st.success(f"### {glyhb:.2f}% (Healthy)")

st.caption("Disclaimer: This tool is for educational purposes only and not a substitute for professional medical advice.")