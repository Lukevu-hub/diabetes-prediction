def user_input_features():
    st.sidebar.header("Nhập thông số bệnh nhân:")
    
    # Nhập các chỉ số cơ bản
    chol = st.sidebar.number_input("Cholesterol", min_value=100, max_value=500, value=200)
    stab_glu = st.sidebar.number_input("Stabilized Glucose", min_value=40, max_value=400, value=100)
    hdl = st.sidebar.number_input("HDL (Good Cholesterol)", min_value=10, max_value=120, value=50)
    age = st.sidebar.slider("Age", 18, 100, 45)
    waist = st.sidebar.slider("Waist (inch)", 20, 60, 35)
    hip = st.sidebar.slider("Hip (inch)", 20, 60, 40)
    
    # Nhập Cân nặng & Chiều cao để tính BMI
    weight_kg = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    height_cm = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)

    # --- LOGIC TỰ ĐỘNG TÍNH TOÁN ---
    
    # 1. Tính Ratio (Tỷ lệ đường huyết trên mỡ tốt)
    # Công thức: Ratio = Glucose / HDL
    ratio = stab_glu / hdl
    
    # 2. Tính BMI (Body Mass Index)
    # Công thức: BMI = weight(kg) / [height(m)]^2
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    # Hiển thị các chỉ số vừa tính được lên màn hình chính để user kiểm tra
    st.sidebar.info(f"💡 Calculated Ratio: {ratio:.2f}")
    st.sidebar.info(f"💡 Calculated BMI: {bmi:.2f}")

    # Tạo DataFrame với ĐÚNG tên cột và THỨ TỰ mà mô hình XGBoost yêu cầu
    data = {
        'chol': chol, 
        'stab.glu': stab_glu, 
        'hdl': hdl, 
        'ratio': ratio,
        'age': age, 
        'waist': waist, 
        'hip': hip, 
        'bmi': bmi
    }
    return pd.DataFrame([data])