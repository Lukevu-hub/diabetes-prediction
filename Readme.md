Diabetes Risk Predictor
==========================

**Disclaimer: This tool is for educational purposes only and not a substitute for professional medical advice.**
-------------------
This project is an end-to-end Machine Learning application designed to predict **Glycated Hemoglobin (Glyhb)** levels---a key indicator of diabetes risk. It includes a complete data science pipeline from Exploratory Data Analysis (EDA) and model training to a production-ready web interface built with **Streamlit**.

🚀 Overview
-------------------

The core objective is to provide a user-friendly tool where individuals can input their physiological data (like cholesterol, glucose, and body measurements) to receive an estimate of their diabetes risk stage.

-   **Target Variable:** `glyhb` (Glycated Hemoglobin).

-   **Classification Logic:**

    -   **Normal:** `glyhb`  < 5.7

    -   **Pre-Diabetes:** 5.7 ≤ `glyhb`  < 6.5

    -   **Diabetes:** `glyhb`  ≥ 6.5

* * * * *

📁 Structure
--------------------

-   `notebook.ipynb`: The development environment containing data cleaning, EDA, feature selection, and model fine-tuning.

-   `app.py`: The Streamlit-based web application providing a graphical user interface for real-time predictions.

-   `diabetes_xgb_model_v1.joblib`: The serialized final XGBoost model used for production.

-   `requirements.txt`: List of Python dependencies.

-   `setup.text`: Bash commands for environment setup and data acquisition.

* * * * *

🛠️ Installation & Setup
------------------------

To run this project locally, follow these steps as defined in the setup configuration^8^:

1.  **Create and activate a virtual environment:**

    ```
    python -m venv venv
    source venv/bin/activate  
    ```

2.  **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

3.  **Download the dataset:**

    The project uses a diabetes dataset from Kaggle.

    ```
    pip install kagglehub
    python download_data.py
    ```

* * * * *

🖥️ Usage
---------

To launch the web application, run the following command in your terminal:

```
streamlit run app.py
```

### **Features of the Web App:**

-   **Unit Converter:** A built-in sidebar tool to convert Imperial units (lbs, inches) to Metric (kg, cm) and Blood Glucose, Cholesterol units (mmol/L to mg/dL).

-   **Input Summary:** Displays a summary of the data you entered, including calculated metrics like BMI.

-   **Risk Analysis:** Predicts the Glyhb value and provides a clear risk category (Normal, Pre-Diabetes, or Diabetes).

* * * * *

📊 Machine Learning Pipeline
----------------------------

### **1\. Data Preprocessing**

-   **Missing Values:** Handled using `SimpleImputer` with a **median** strategy for numeric features and **mode** for categorical features.

-   **Scaling:** Features are normalized using `StandardScaler` to ensure optimal performance for algorithms like Ridge and XGBoost.

### **2\. Model Selection & Performance**

Multiple models were evaluated using Root Mean Squared Error (RMSE) and R-squared metrics:

| **Model** | **RMSE** | **R-squared** |
| --- | --- | --- |
| Ridge Regression | 0.9835 | 0.8201 |
| Random Forest | 1.3520 | 0.6600 |
| **XGBoost (Selected)** | **1.4157** | **0.6272** |

Note: While Ridge showed a lower RMSE in initial tests, the project focuses on the **XGBoost** model for final deployment due to its better performance after fine-tuning and ability to handle non-linear relationships.

### **3\. Model Export**

The final model is exported as a `.joblib` file using a Scikit-learn Pipeline, which ensures that preprocessing (imputation and scaling) is bundled together with the predictor for consistency.

* * * * *

📦 Requirements
---------------

The project relies on the following libraries:

-   `pandas` & `numpy`: For data manipulation.

-   `matplotlib` & `seaborn`: For visualization

-   `scikit-learn`: For the ML pipeline and preprocessing.

-   `xgboost`: The core prediction algorithm.

-   `joblib`: For model serialization.

-   `streamlit`: For the web interface.

* * * * *

📝 Author
---------

Luke Vu

Current Focus: Data Science and Applied ML Engineering.
