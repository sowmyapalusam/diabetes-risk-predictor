# Early Diabetes Risk Prediction Using Machine Learning

A Streamlit prototype that uses Random Forest and Decision Tree classifiers to estimate diabetes risk from basic health parameters.

## Features
- Glucose, BMI, age, blood pressure and other health inputs
- Missing-value handling with median imputation
- Random Forest and Decision Tree models
- Instant web-based prediction
- Test-set accuracy shown for transparency
- Clear medical disclaimer

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
The prototype uses the Pima Indians Diabetes dataset hosted in a public GitHub repository:
https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset

The application is an educational screening prototype and is not a medical diagnosis.
