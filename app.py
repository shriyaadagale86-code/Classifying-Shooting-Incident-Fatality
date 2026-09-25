import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="NYPD Shooting Incident Outcome Predictor", layout="centered"
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, label_encoders, feature_columns


model, scaler, label_encoders, feature_columns = load_artifacts()

st.title("NYPD Shooting Incident Fatality Predictor")
st.write(
    "Predict whether a shooting incident resulted in a fatality (`STATISTICAL_MURDER_FLAG`)."
)

st.subheader("Incident Details")

# Input widgets for key attributes
input_data = {}

col1, col2 = st.columns(2)

with col1:
    input_data["BORO"] = st.selectbox("Borough", label_encoders["BORO"].classes_)
    input_data["PRECINCT"] = st.number_input(
        "Precinct", min_value=1, max_value=123, value=40
    )
    input_data["JURISDICTION_CODE"] = st.number_input(
        "Jurisdiction Code", min_value=0.0, max_value=2.0, value=0.0
    )
    input_data["LOCATION_DESC"] = st.selectbox(
        "Location Description", label_encoders["LOCATION_DESC"].classes_
    )
    input_data["LOC_CLASSFCTN_DESC"] = st.selectbox(
        "Location Classification", label_encoders["LOC_CLASSFCTN_DESC"].classes_
    )
    input_data["PERP_AGE_GROUP"] = st.selectbox(
        "Perpetrator Age Group", label_encoders["PERP_AGE_GROUP"].classes_
    )
    input_data["PERP_SEX"] = st.selectbox(
        "Perpetrator Sex", label_encoders["PERP_SEX"].classes_
    )
    input_data["PERP_RACE"] = st.selectbox(
        "Perpetrator Race", label_encoders["PERP_RACE"].classes_
    )
    input_data["OCCUR_HOUR"] = st.slider(
        "Occur Hour (0-23)", min_value=0, max_value=23, value=12
    )

with col2:
    input_data["VIC_AGE_GROUP"] = st.selectbox(
        "Victim Age Group", label_encoders["VIC_AGE_GROUP"].classes_
    )
    input_data["VIC_SEX"] = st.selectbox(
        "Victim Sex", label_encoders["VIC_SEX"].classes_
    )
    input_data["VIC_RACE"] = st.selectbox(
        "Victim Race", label_encoders["VIC_RACE"].classes_
    )
    input_data["X_COORD_CD"] = st.number_input("X Coordinate", value=1005028.0)
    input_data["Y_COORD_CD"] = st.number_input("Y Coordinate", value=234516.0)
    input_data["Latitude"] = st.number_input("Latitude", value=40.810352)
    input_data["Longitude"] = st.number_input("Longitude", value=-73.924942)
    input_data["Lon_Lat"] = st.selectbox(
        "Lon_Lat String", label_encoders["Lon_Lat"].classes_
    )

# Prediction execution
if st.button("Predict Outcome"):
    input_df = pd.DataFrame([input_data])

    # Encode categorical values
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col].astype(str))

    # Align column order
    input_df = input_df[feature_columns]

    # Scale inputs
    scaled_inputs = scaler.transform(input_df)

    # Inference
    prediction = model.predict(scaled_inputs)[0]
    probability = model.predict_proba(scaled_inputs)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(
            f"Result: Fatal Outcome Predicted (Probability: {probability:.2%})"
        )
    else:
        st.success(
            f"Result: Non-Fatal Outcome Predicted (Probability of Fatality: {probability:.2%})"
        )
