
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Customer Churn Prediction", layout="wide")

st.title("Prediksi Customer Churn Pelanggan Telco")
st.write("Aplikasi sederhana untuk memprediksi apakah pelanggan berpotensi churn atau tidak.")

MODEL_PATH = "telco_churn_best_model.pkl"
SCHEMA_PATH = "feature_schema.json"

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

st.subheader("Input Data Pelanggan")

input_data = {}

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Fitur Numerik")
    for col in schema["numerical_features"]:
        min_value = schema["numerical_min"][col]
        max_value = schema["numerical_max"][col]
        default_value = schema["numerical_defaults"][col]

        if col == "SeniorCitizen":
            input_data[col] = st.selectbox(
                col,
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        else:
            input_data[col] = st.number_input(
                col,
                min_value=float(min_value),
                max_value=float(max_value),
                value=float(default_value)
            )

with col2:
    st.markdown("### Fitur Kategorikal")
    for col in schema["categorical_features"]:
        options = schema["categorical_options"][col]
        input_data[col] = st.selectbox(col, options=options)

if st.button("Prediksi Churn"):
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.error("Pelanggan diprediksi CHURN")
    else:
        st.success("Pelanggan diprediksi TIDAK CHURN")

    st.write(f"Probabilitas Churn: {probability:.2%}")

    st.markdown("### Interpretasi")
    if probability >= 0.7:
        st.write("Risiko churn tinggi. Pelanggan perlu menjadi prioritas retensi.")
    elif probability >= 0.4:
        st.write("Risiko churn sedang. Pelanggan perlu dipantau dan dapat diberikan promo/penawaran.")
    else:
        st.write("Risiko churn rendah. Pelanggan relatif aman.")
