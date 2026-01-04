import streamlit as st
import joblib
import pandas as pd
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# --- HÀM NHẬP LIỆU ---
def user_input_features():
    st.sidebar.header("Nhập thông số bệnh nhân:")
    
    chol = st.sidebar.number_input("Cholesterol", min_value=100, max_value=500, value=200)
    stab_glu = st.sidebar.number_input("Stabilized Glucose", min_value=40, max_value=400, value=100)
    hdl = st.sidebar.number_input("HDL (Good Cholesterol)", min_value=10, max_value=120, value=50)
    age = st.sidebar.slider("Age", 18, 100, 45)
    waist = st.sidebar.slider("Waist (inch)", 20, 60, 35)
    hip = st.sidebar.slider("Hip (inch)", 20, 60, 40)
    
    weight_kg = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    height_cm = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)

    # Logic tự động tính toán
    ratio = stab_glu / hdl
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    # Hiển thị các chỉ số vừa tính được lên màn hình chính để user kiểm tra
    st.sidebar.info(f"💡 Calculated Ratio: {ratio:.2f}")
    st.sidebar.info(f"💡 Calculated BMI: {bmi:.2f}")

    data = {
        'chol': chol, 'stab.glu': stab_glu, 'hdl': hdl, 'ratio': ratio,
        'age': age, 'waist': waist, 'hip': hip, 'bmi': bmi
    }
    return pd.DataFrame([data])

# --- GIAO DIỆN CHÍNH ---
st.title("🩺 Diabetes Risk Prediction")
st.write("Dự án nghiên cứu AI/ML - Luke Vu")

# 1. Kiểm tra File Model
model_path = 'diabetes_xgb_model_v1.joblib'

if not os.path.exists(model_path):
    st.error(f"❌ KHÔNG tìm thấy file: {model_path}")
    st.write("Các file hiện có trong thư mục này là:", os.listdir('.'))
else:
    # 2. Tải mô hình
    try:
        model = joblib.load(model_path)
        st.success("🚀 Mô hình đã được nạp thành công!")
        
        # 3. Lấy dữ liệu người dùng
        input_df = user_input_features()
        
        st.subheader("📋 Thông số đã nhập")
        st.write(input_df)

        # 4. Dự đoán
        if st.button("Dự đoán kết quả"):
            prediction = model.predict(input_df)
            result = prediction[0]
            
            st.markdown("---")
            st.header(f"Kết quả dự đoán Glyhb: {result:.2f}")
            
            if result >= 6.5:
                st.error("⚠️ Trạng thái: Nguy cơ Tiểu đường cao")
            elif result >= 5.7:
                st.warning("🟠 Trạng thái: Tiền tiểu đường")
            else:
                st.success("✅ Trạng thái: Bình thường")
                
    except Exception as e:
        st.error(f"⚠️ Lỗi khi chạy mô hình: {e}")
