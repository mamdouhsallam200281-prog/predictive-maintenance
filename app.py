# app.py
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predictive Maintenance AI", page_icon="⚙️", layout="wide")

model = joblib.load("best_predictive_maintenance_model.joblib")

st.title("⚙️ Predictive Maintenance AI")
st.write("Enter machine operating conditions to predict whether a failure is likely.")

col1, col2, col3 = st.columns(3)
with col1:
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
    air_temp = st.number_input("Air temperature [K]", value=298.0, step=0.1)
with col2:
    process_temp = st.number_input("Process temperature [K]", value=308.0, step=0.1)
    speed = st.number_input("Rotational speed [rpm]", value=1500, step=10)
with col3:
    torque = st.number_input("Torque [Nm]", value=40.0, step=0.1)
    tool_wear = st.number_input("Tool wear [min]", value=100, step=1)

type_mapping = {"L": 0, "M": 1, "H": 2}
input_data = pd.DataFrame([{
    "Type": type_mapping[machine_type],
    "Air_Temperature_K": air_temp,
    "Process_Temperature_K": process_temp,
    "Rotational_Speed_rpm": speed,
    "Torque_Nm": torque,
    "Tool wear [min]": tool_wear
}])

if st.button("Predict Failure", type="primary"):
    prediction = int(model.predict(input_data.values)[0])
    st.divider()
    if prediction == 1:
        st.error("⚠️ Failure predicted")
    else:
        st.success("✅ No failure predicted")

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data.values)[0, 1])
        st.metric("Estimated failure probability", f"{probability:.1%}")

st.caption("Educational project — prediction is based on the trained dataset model.")
