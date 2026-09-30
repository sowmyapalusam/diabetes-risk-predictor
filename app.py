"""Streamlit user interface for the Early Diabetes Risk Predictor."""
from __future__ import annotations

import streamlit as st

from model import FEATURE_NAMES, predict_risk, train_models

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Early Diabetes Risk Predictor")
st.caption(
    "Machine-learning screening prototype using basic health parameters. "
    "It is not a medical diagnosis."
)

@st.cache_resource(show_spinner=False)
def get_models():
    """Train and cache the two classifiers used by the application."""
    return train_models()

try:
    random_forest, decision_tree, metrics, importance = get_models()
except Exception as exc:
    st.error(f"The model could not be initialized: {exc}")
    st.caption("Check the deployment logs and network access to the public dataset.")
    st.stop()

st.sidebar.header("Patient Inputs")
st.sidebar.caption("Enter values available from a routine health assessment.")

values = {}
values["Pregnancies"] = st.sidebar.number_input(
    "Pregnancies", min_value=0.0, max_value=20.0, value=1.0, step=1.0
)
values["Glucose"] = st.sidebar.number_input(
    "Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0
)
values["BloodPressure"] = st.sidebar.number_input(
    "Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0
)
values["SkinThickness"] = st.sidebar.number_input(
    "Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0
)
values["Insulin"] = st.sidebar.number_input(
    "Insulin (µU/mL)", min_value=0.0, max_value=900.0, value=80.0, step=1.0
)
values["BMI"] = st.sidebar.number_input(
    "BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1
)
values["DiabetesPedigreeFunction"] = st.sidebar.number_input(
    "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01
)
values["Age"] = st.sidebar.number_input(
    "Age (years)", min_value=1.0, max_value=120.0, value=30.0, step=1.0
)

if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
    result = predict_risk(random_forest, values)

    st.subheader("Prediction")
    probability = result["probability"]

    if result["risk"] == "Higher predicted risk":
        st.warning(f"⚠️ {result['risk']} — model probability: {probability:.1%}")
    else:
        st.success(f"✅ {result['risk']} — model probability: {probability:.1%}")

    st.info(
        "This result is a screening estimate from a machine-learning model. "
        "It should not be used to diagnose diabetes or replace professional medical advice."
    )

with st.expander("Model validation"):
    st.write(
        f"Random Forest accuracy: **{metrics['random_forest_accuracy']:.1%}**"
    )
    st.write(
        f"Decision Tree accuracy: **{metrics['decision_tree_accuracy']:.1%}**"
    )
    st.write(
        f"Validation samples: **{metrics['validation_samples']}**"
    )

with st.expander("Key model features"):
    for feature, score in list(importance.items())[:5]:
        st.write(f"**{feature}** — {score:.3f}")

st.divider()
st.caption(
    "Dataset: Pima Indians Diabetes dataset. Values such as zero measurements "
    "for selected clinical fields are treated as missing during preprocessing."
)
