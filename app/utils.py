import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

from constants import (CATEGORICAL_MAPPINGS, ELEVATED_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
                       MODEL_PATH, SCALER_PATH, X_TRAIN_PATH)

@st.cache_resource(show_spinner="Loading model and resources...")
def load_resources():
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    X_train = joblib.load(X_TRAIN_PATH)
    feature_names = X_train.columns.tolist()
    feature_defaults = {}

    for column in feature_names:
        if X_train[column].nunique() <= 10:
            feature_defaults[column] = X_train[column].mode().iloc[0]
        else:
            feature_defaults[column] = X_train[column].median()

    dummy_input = np.zeros((1, len(feature_names)), dtype=np.float32)

    model(dummy_input, training=False)

    return (model, scaler, feature_names, feature_defaults)

def binary_label(value: int) -> str:
    return "Yes" if value == 1 else "No"

def build_application_data(
    payment_type,
    employment_status,
    housing_status,
    source,
    device_os,
    income,
    name_email_similarity,
    customer_age,
    days_since_request,
    date_of_birth_distinct_emails_4w,
    credit_risk_score,
    email_is_free,
    phone_home_valid,
    phone_mobile_valid,
    has_other_cards,
    proposed_credit_limit,
    foreign_request,
    keep_alive_session,
    device_distinct_emails_8w,
    device_fraud_count,
    month) -> dict:
    return {
        "payment_type": payment_type,
        "employment_status": employment_status,
        "housing_status": housing_status,
        "source": source,
        "device_os": device_os,
        "income": income,
        "name_email_similarity": name_email_similarity,
        "customer_age": customer_age,
        "days_since_request": days_since_request,
        "date_of_birth_distinct_emails_4w": (date_of_birth_distinct_emails_4w),
        "credit_risk_score": credit_risk_score,
        "email_is_free": email_is_free,
        "phone_home_valid": phone_home_valid,
        "phone_mobile_valid": phone_mobile_valid,
        "has_other_cards": has_other_cards,
        "proposed_credit_limit": proposed_credit_limit,
        "foreign_request": foreign_request,
        "keep_alive_session": keep_alive_session,
        "device_distinct_emails_8w": device_distinct_emails_8w,
        "device_fraud_count": device_fraud_count,
        "month": month,
    }

def prepare_input(application_data: dict) -> np.ndarray:
    (_, scaler, feature_names, feature_defaults) = load_resources()

    complete_data = feature_defaults.copy()
    complete_data.update(application_data)

    input_df = pd.DataFrame([complete_data])

    for column, mapping in CATEGORICAL_MAPPINGS.items():
        value = input_df[column].iloc[0]

        if value not in mapping:
            raise ValueError(f"Invalid value '{value}' for '{column}'.")

        input_df[column] = input_df[column].map(mapping).astype("int64")

    input_df = input_df[feature_names].astype(np.float32)
    input_scaled = scaler.transform(input_df).astype(np.float32)

    return input_scaled

def predict_fraud(application_data: dict) -> float:
    (model, _, _, _,) = load_resources()

    input_scaled = prepare_input(application_data)

    prediction = model(input_scaled, training=False)

    return float(prediction.numpy()[0][0])

def get_risk_assessment(risk_score: float) -> dict:
    if risk_score >= HIGH_RISK_THRESHOLD:
        return {
            "level": "High Risk",
            "message": ("Manual fraud investigation is recommended before approving the application.")
        }

    if risk_score >= ELEVATED_RISK_THRESHOLD:
        return {
            "level": "Elevated Risk",
            "message": ("Additional identity and application verification is recommended.")
        }

    return {
        "level": "Low Risk",
        "message": ("No strong fraud signal was identified. Standard verification procedures should still be followed.")
    }