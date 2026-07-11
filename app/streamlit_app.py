import streamlit_app as st
from constants import (
    APPLICATION_SOURCES,
    BINARY_OPTIONS,
    CUSTOMER_AGE_OPTIONS,
    DEVICE_DISTINCT_EMAIL_OPTIONS,
    DEVICE_OS_OPTIONS,
    EMPLOYMENT_STATUSES,
    GITHUB_URL,
    HOUSING_STATUSES,
    INCOME_OPTIONS,
    LINKEDIN_URL,
    MONTH_OPTIONS,
    PAYMENT_TYPES,
    PROPOSED_CREDIT_LIMIT_OPTIONS)

from utils import (binary_label, build_application_data, get_risk_assessment, load_resources, predict_fraud)

st.set_page_config(page_title="Bank Account Opening Fraud Detection", page_icon="🏦", layout="wide")

load_resources()

st.markdown(
    """
    <div style="background-color:#0E1117;padding:25px;border-radius:12px;margin-bottom:25px">
        <h1 style="color:white;margin-bottom:5px;">
            🏦 AI Bank Account Opening Fraud Detection
        </h1>
        <p style="color:#D3D3D3;font-size:18px;margin-bottom:0;">
            Deep Learning decision-support application for identifying
            potentially fraudulent bank account opening applications.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info("This application supports fraud review prioritisation. It should not be used as an automatic approval or rejection system.")

with st.form("fraud_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Applicant Details")

        income = st.selectbox("Income Level", INCOME_OPTIONS, index=5)

        customer_age = st.selectbox("Customer Age", CUSTOMER_AGE_OPTIONS, index=2)

        employment_status = st.selectbox("Employment Status", EMPLOYMENT_STATUSES)

        housing_status = st.selectbox("Housing Status", HOUSING_STATUSES)

        name_email_similarity = st.slider("Name and Email Similarity", min_value=0.0, max_value=1.0, value=0.70, step=0.01)

        date_of_birth_distinct_emails_4w = st.number_input("Distinct Emails Associated with Date of Birth",
                                                           min_value=0, max_value=39, value=7, step=1)

        email_is_free = st.selectbox("Free Email Provider", BINARY_OPTIONS, format_func=binary_label)

        phone_home_valid = st.selectbox("Valid Home Phone", BINARY_OPTIONS, format_func=binary_label)

        phone_mobile_valid = st.selectbox("Valid Mobile Phone", BINARY_OPTIONS, index=1, format_func=binary_label)

        has_other_cards = st.selectbox("Has Other Bank Cards", BINARY_OPTIONS, format_func=binary_label)

    with col2:
        st.subheader("💳 Application & Device Details")

        payment_type = st.selectbox("Payment Type", PAYMENT_TYPES)

        source = st.selectbox("Application Source", APPLICATION_SOURCES)

        device_os = st.selectbox("Device Operating System", DEVICE_OS_OPTIONS)

        days_since_request = st.number_input("Days Since Request", min_value=0.0, value=0.01, format="%.6f")

        credit_risk_score = st.number_input("Credit Risk Score", min_value=-170, max_value=387, value=110, step=1)

        proposed_credit_limit = st.selectbox("Proposed Credit Limit", PROPOSED_CREDIT_LIMIT_OPTIONS, index=1)

        foreign_request = st.selectbox("Foreign Request", BINARY_OPTIONS, format_func=binary_label)

        keep_alive_session = st.selectbox("Keep Alive Session", BINARY_OPTIONS, index=1, format_func=binary_label)

        device_distinct_emails_8w = st.selectbox("Distinct Emails Associated with Device", DEVICE_DISTINCT_EMAIL_OPTIONS, index=2)

        device_fraud_count = st.selectbox("Previous Device Fraud Count", [0])

        month = st.selectbox("Application Month", MONTH_OPTIONS)

    submitted = st.form_submit_button("Analyse Fraud Risk", use_container_width=True)

if submitted:
    application_data = build_application_data(
        payment_type=payment_type,
        employment_status=employment_status,
        housing_status=housing_status,
        source=source,
        device_os=device_os,
        income=income,
        name_email_similarity=name_email_similarity,
        customer_age=customer_age,
        days_since_request=days_since_request,
        date_of_birth_distinct_emails_4w=(date_of_birth_distinct_emails_4w),
        credit_risk_score=credit_risk_score,
        email_is_free=email_is_free,
        phone_home_valid=phone_home_valid,
        phone_mobile_valid=phone_mobile_valid,
        has_other_cards=has_other_cards,
        proposed_credit_limit=proposed_credit_limit,
        foreign_request=foreign_request,
        keep_alive_session=keep_alive_session,
        device_distinct_emails_8w=device_distinct_emails_8w,
        device_fraud_count=device_fraud_count,
        month=month)

    try:
        with st.spinner("Analysing application..."):
            risk_score = predict_fraud(application_data)

        assessment = get_risk_assessment(risk_score)

        st.markdown("---")
        st.subheader("🎯 Fraud Risk Assessment")

        metric1, metric2 = st.columns(2)
        metric1.metric("Fraud Risk Score", f"{risk_score:.1%}")
        metric2.metric("Risk Level", assessment["level"])

        st.progress(min(max(risk_score, 0.0), 1.0))

        if assessment["level"] == "High Risk":
            st.error(f"🔴 **High Risk**\n\n{assessment['message']}")

        elif assessment["level"] == "Elevated Risk":
            st.warning(f"🟠 **Elevated Risk**\n\n{assessment['message']}")

        else:
            st.success(f"🟢 **Low Risk**\n\n{assessment['message']}")

    except Exception as error:
        st.error("The application could not generate a prediction.")
        st.exception(error)

st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center;color:gray">
        Developed by Mauricio Mazuera ·
        <a href="{GITHUB_URL}" target="_blank">GitHub</a> ·
        <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)