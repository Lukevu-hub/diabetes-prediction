import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

# --- 1. Optimization: Load Model ---
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

model_path = 'diabetes_xgb_model_v1.joblib'
model = load_model(model_path)

# --- 2. Unit Converter ---
def convert_units():
    with st.sidebar.expander("🔄 Unit Converter", expanded=False):        
        # Weight & Height
        lbs = st.number_input("Weight (lbs)", value=154.0)
        st.info(f"**{lbs / 2.2046:.2f} kg**")
    
        inches = st.number_input("Length (inch)", value=34.0)
        st.info(f"**{inches * 2.54:.2f} cm**")  
        # Blood test values (mmol/L -> mg/dL)
        mmol = st.number_input("Value (mmol/L)", value=5.5)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Glucose")
            st.info(f"**{mmol * 18.018:.1f} mg/dL**")
        with c2:
            st.caption("Cholesterol")
            st.info(f"**{mmol * 38.67:.1f} mg/dL**")

# --- 3. Input Interface ---
def get_user_inputs():
    st.sidebar.header("📋 Your Information")
    
    convert_units() # Call Unit Converter
    
    with st.sidebar.expander("🩸 Blood Test Results", expanded=True):
        chol = st.number_input("Total Cholesterol (mg/dL)", 100, 500, 200)
        hdl = st.number_input("HDL (mg/dL)", 10, 120, 50)
        stab_glu = st.number_input("Stabilized Glucose (mg/dL)", 40, 400, 100)
        
    with st.sidebar.expander("📏 Body Measurements", expanded=True):
        age = st.slider("Age", 18, 100, 45)
        waist = st.slider("Waist (cm)", 20, 150, 80)
        hip = st.slider("Hip (cm)", 20, 150, 95)
        weight = st.slider("Weight (kg)", 30, 200, 70)
        height = st.slider("Height (cm)", 100, 250, 170)

    # Feature Engineering 
    ratio = stab_glu / hdl
    bmi = weight / ((height/100) ** 2)
    
    data = {
        'chol': chol, 'stab.glu': stab_glu, 'hdl': hdl, 'ratio': ratio,
        'age': age, 'waist': waist, 'hip': hip, 'bmi': bmi
    }
    return pd.DataFrame([data])

# --- 4. Main Dashboard ---
st.title("🩺 Diabetes Risk Prediction")
st.markdown(f"**Developer:** Luke Vu  |  **Target:** Applied ML Engineer")
st.warning("Disclaimer: This tool is for educational purposes only and not a substitute for professional medical advice.")


st.divider()

if model is None:
    st.error(f"❌ Can not find file model: `{model_path}`")
    st.stop()

# --- 4. Main Dashboard ---
input_df = get_user_inputs()

# --- AGE GROUP ---
age_bins = [0, 20, 30, 40, 50, 60, 100]
age_labels = ["<20", "21-30", "31-40", "41-50", "51-60", "60+"]

# Tính toán age_group để hiển thị, KHÔNG gán vào input_df
current_age = input_df['age'].iloc[0]
current_age_group = pd.cut([current_age], bins=age_bins, labels=age_labels)[0]

# Metrics Display
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Calculated BMI", f"{input_df['bmi'].iloc[0]:.2f}")
with col2:
    st.metric("Glucose/HDL Ratio", f"{input_df['ratio'].iloc[0]:.2f}")
with col3:
    # Show Age Group
    st.metric("Age Group", value=str(current_age_group))

st.subheader("📋 Input Summary")
# Display dataframe (only 8 columns) for debugging if needed
st.dataframe(input_df, use_container_width=True)

# --- 5. Prediction ---
if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
    prediction = model.predict(input_df)
    glyhb = prediction[0]
    
    st.divider()
    st.subheader("Predicted Glycated Hemoglobin (Glyhb):")
    st.info("💡 **Note:** Glyhb (HbA1c) represents average blood sugar levels over 2-3 months.")
    
    # Classification based on Glyhb value
    if glyhb >= 6.5:
        st.error(f"### Result: {glyhb:.2f}% (Diabetes Risk)")
    elif glyhb >= 5.7:
        st.warning(f"### Result: {glyhb:.2f}% (Pre-diabetes)")
    else:
        st.success(f"### Result: {glyhb:.2f}% (Healthy / Normal)")
    st.divider()
# Reference Unit Conversion Table displayed in Main Area
with st.expander("📚 Reference Unit Conversion Table"):
    st.markdown("""
    | Measurement | From Unit | Formula | To Unit (Model) |
    | :--- | :--- | :--- | :--- |
    | **Glucose** | 1 mmol/L | x 18.018 | mg/dL |
    | **Cholesterol** | 1 mmol/L | x 38.67 | mg/dL |
    | **Weight** | 1 lb (pound) | / 2.2046 | kg |
    | **Length** | 1 inch | x 2.54 | cm |
    """)