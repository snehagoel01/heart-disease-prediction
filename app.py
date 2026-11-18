
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load trained KNN model
# -------------------------------
data = joblib.load("heart_disease_model.pkl")

model = data["model"]
imputer = data["imputer"]
scaler = data["scaler"]
features = data["features"]

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------------------
# Minimal CSS Design
# -------------------------------
st.markdown("""
<style>
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #183B56;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        color: #64748b;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        color: #183B56;
        font-size: 21px;
        font-weight: 600;
        margin-top: 15px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        height: 45px;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }

    [data-testid="stForm"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 22px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Page Heading
# -------------------------------
st.markdown(
    '<div class="main-title">Heart Disease Prediction</div>',
    unsafe_allow_html=True
)



# -------------------------------
# Patient Input Form
# -------------------------------
st.markdown(
    '<div class="section-title">Patient Information</div>',
    unsafe_allow_html=True
)

with st.form("heart_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Basic Details**")

        age = st.number_input(
            "Age (years)", 1, 120, 45
        )

        sex = st.selectbox(
            "Gender", ["Male", "Female"]
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical Angina",
                2: "Atypical Angina",
                3: "Non-anginal Pain",
                4: "Asymptomatic"
            }[x]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            50, 250, 120
        )

        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            50, 700, 200
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            ["No", "Yes"]
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"
            }[x]
        )

    with col2:
        st.markdown("**Additional Clinical Details**")

        thalach = st.number_input(
            "Maximum Heart Rate Achieved",
            50, 250, 150
        )

        exang = st.selectbox(
            "Exercise-Induced Angina",
            ["No", "Yes"]
        )

        oldpeak = st.number_input(
            "Oldpeak (ST Depression)",
            0.0, 10.0, 1.0, step=0.1
        )

        slope = st.selectbox(
            "ST Segment Slope",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Upsloping",
                2: "Flat",
                3: "Downsloping"
            }[x]
        )

        ca = st.selectbox(
            "Number of Major Vessels (0–3)",
            [0, 1, 2, 3]
        )

        thal = st.selectbox(
            "Thalassemia",
            [3, 6, 7],
            format_func=lambda x: {
                3: "Normal",
                6: "Fixed Defect",
                7: "Reversible Defect"
            }[x]
        )

    st.markdown("---")

    submit = st.form_submit_button(
        "Predict Heart Disease",
        use_container_width=True
    )

# -------------------------------
# Prediction
# -------------------------------
if submit:

    patient = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": restecg,
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    input_df = pd.DataFrame(
        [[patient[f] for f in features]],
        columns=features
    )

    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)

    prediction = model.predict(input_scaled)[0]

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:
        st.error(
            "**Model Prediction: Class 1**\n\n"
            "The model predicts the presence of heart disease."
        )
    else:
        st.success(
            "**Model Prediction: Class 0**\n\n"
            "The model predicts no heart disease."
        )

    st.caption(
        "This is an experimental prediction from a trained KNN model. "
        "It cannot confirm or rule out heart disease. "
        "Please consult a qualified healthcare professional."
    )


