import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

DATA_URL = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

@st.cache_resource
def train_models():
    df = pd.read_csv(DATA_URL)

    # In this dataset, zero values in these measurements are treated as missing.
    missing_as_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[missing_as_zero] = df[missing_as_zero].replace(0, pd.NA)

    X = df[FEATURES]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")

    rf = Pipeline([
        ("imputer", imputer),
        ("model", RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced"
        ))
    ])

    dt = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            random_state=42, max_depth=5, class_weight="balanced"
        ))
    ])

    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    dt_acc = accuracy_score(y_test, dt.predict(X_test))

    return rf, dt, rf_acc, dt_acc

st.title("🩺 Early Diabetes Risk Predictor")
st.write("A machine-learning prototype that estimates diabetes risk from basic health parameters.")

with st.spinner("Preparing the machine-learning models..."):
    rf, dt, rf_acc, dt_acc = train_models()

st.sidebar.header("Enter Health Details")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose (mg/dL)", min_value=1.0, max_value=300.0, value=120.0)
blood_pressure = st.sidebar.number_input("Blood Pressure (mmHg)", min_value=1.0, max_value=200.0, value=70.0)
skin_thickness = st.sidebar.number_input("Skin Thickness (mm)", min_value=1.0, max_value=100.0, value=20.0)
insulin = st.sidebar.number_input("Insulin (µU/mL)", min_value=1.0, max_value=900.0, value=80.0)
bmi = st.sidebar.number_input("BMI", min_value=1.0, max_value=70.0, value=25.0)
pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47)
age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=30)

input_df = pd.DataFrame([{
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": pedigree,
    "Age": age
}])

st.subheader("Prediction")

if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
    probability = rf.predict_proba(input_df)[0][1]
    prediction = int(rf.predict(input_df)[0])

    if prediction == 1:
        st.warning(f"⚠️ Higher predicted risk ({probability:.1%})")
        st.write("The model identifies a higher-risk pattern in the entered values.")
    else:
        st.success(f"✅ Lower predicted risk ({probability:.1%})")
        st.write("The model identifies a lower-risk pattern in the entered values.")

    st.info("This is a machine-learning screening prototype, not a medical diagnosis. Consult a qualified healthcare professional for clinical evaluation.")

with st.expander("Model information"):
    st.write(f"Random Forest test accuracy: **{rf_acc:.1%}**")
    st.write(f"Decision Tree test accuracy: **{dt_acc:.1%}**")
    st.write("The models use the Pima Indians Diabetes dataset and median imputation for missing values. Results depend on the dataset and should not be interpreted as clinical accuracy.")
