import streamlit as st
import pandas as pd
import joblib
import os

# --- Cấu hình trang ---
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

# --- 1. Tối ưu hóa: Load Model ---
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

model_path = 'diabetes_xgb_model_v1.joblib'
model = load_model(model_path)

# --- 2. Bộ chuyển đổi đơn vị (Helper Function) ---
def convert_units():
    with st.sidebar.expander("🔄 Unit Converter", expanded=False):        
        # Cân nặng & Chiều cao
        lbs = st.number_input("Weight (lbs)", value=154.0)
        st.info(f"**{lbs / 2.2046:.2f} kg**")
    
        inches = st.number_input("Length (inch)", value=34.0)
        st.info(f"**{inches * 2.54:.2f} cm**")  
        # Chỉ số máu (mmol/L -> mg/dL)
        mmol = st.number_input("Value (mmol/L)", value=5.5)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Glucose")
            st.code(f"{mmol * 18.018:.1f}")
        with c2:
            st.caption("Cholesterol")
            st.code(f"{mmol * 38.67:.1f}")

# --- 3. Giao diện nhập liệu ---
def get_user_inputs():
    st.sidebar.header("📋 Patient Information")
    
    convert_units() # Gọi bộ chuyển đổi
    
    with st.sidebar.expander("🩸 Blood Test Results", expanded=True):
        chol = st.number_input("Total Cholesterol (mg/dL)", 100, 500, 200)
        hdl = st.number_input("HDL (mg/dL)", 10, 120, 50)
        stab_glu = st.number_input("Stabilized Glucose (mg/dL)", 40, 400, 100)
        
    with st.sidebar.expander("📏 Body Measurements", expanded=True):
        age = st.slider("Age", 18, 100, 45)
        waist = st.slider("Waist (cm)", 20, 150, 80)
        hip = st.slider("Hip (cm)", 20, 150, 95)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)

    # Feature Engineering (Tạo thêm biến cho mô hình)
    ratio = stab_glu / hdl
    bmi = weight / ((height/100) ** 2)
    
    data = {
        'chol': chol, 'stab.glu': stab_glu, 'hdl': hdl, 'ratio': ratio,
        'age': age, 'waist': waist, 'hip': hip, 'bmi': bmi
    }
    return pd.DataFrame([data])

# --- 4. Main Dashboard ---
st.title("🩺 Diabetes Risk Prediction")
st.markdown(f"**Developer:** Luke Vu | **Target:** Applied ML Engineer")

# Bảng tra cứu hiển thị trong Main Area
with st.expander("📚 Reference Unit Conversion Table"):
    st.markdown("""
    | Measurement | From Unit | Formula | To Unit (Model) |
    | :--- | :--- | :--- | :--- |
    | **Glucose** | 1 mmol/L | x 18.018 | mg/dL |
    | **Cholesterol** | 1 mmol/L | x 38.67 | mg/dL |
    | **Weight** | 1 lb (pound) | / 2.2046 | kg |
    | **Length** | 1 inch | x 2.54 | cm |
    """)

st.divider()

if model is None:
    st.error(f"❌ Không tìm thấy file model: `{model_path}`")
    st.stop()

input_df = get_user_inputs()

# Hiển thị Metrics chính
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Calculated BMI", f"{input_df['bmi'].iloc[0]:.2f}")
with col2:
    st.metric("Glucose/HDL Ratio", f"{input_df['ratio'].iloc[0]:.2f}")
with col3:
    st.metric("Age Group", f"{input_df['age'].iloc[0]} yrs")

st.subheader("📋 Input Summary")
st.dataframe(input_df, use_container_width=True)

# --- 5. Logic Dự Đoán (Prediction) ---
if st.button("Analyze Risk Level", type="primary", use_container_width=True):
    prediction = model.predict(input_df)
    glyhb = prediction[0]
    
    st.divider()
    st.subheader("Predicted Glycated Hemoglobin (Glyhb):")
    st.info("💡 **Note:** Glyhb (HbA1c) represents average blood sugar levels over 2-3 months.")
    
    # Logic phân loại theo tiêu chuẩn y tế
    if glyhb >= 6.5:
        st.error(f"### Result: {glyhb:.2f}% (Diabetes Risk)")
    elif glyhb >= 5.7:
        st.warning(f"### Result: {glyhb:.2f}% (Pre-diabetes)")
    else:
        st.success(f"### Result: {glyhb:.2f}% (Healthy / Normal)")

st.divider()
st.caption("Disclaimer: This tool is for educational purposes only and not a substitute for professional medical advice.")