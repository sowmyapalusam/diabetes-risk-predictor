# Early Diabetes Risk Prediction Using Machine Learning

## Problem
Diabetes can develop without obvious symptoms. This project provides a lightweight machine-learning screening prototype that estimates risk from common health parameters.

## Methodology
1. Load the Pima Indians Diabetes dataset.
2. Treat impossible zero measurements in selected clinical fields as missing.
3. Split data using a fixed, stratified 80/20 train-validation split.
4. Apply median imputation inside a scikit-learn Pipeline to prevent validation leakage.
5. Train and compare Random Forest and Decision Tree classifiers.
6. Expose the Random Forest prediction through a Streamlit interface.
7. Display validation accuracy, probability, and model feature importance.

## Architecture
`Streamlit UI -> input validation -> cached ML model -> preprocessing pipeline -> prediction -> risk explanation`

## Engineering improvements
- PEP-484 type hints on core functions.
- Structured docstrings and validation.
- Automated pytest tests for missing values, invalid inputs, prediction contract, and required fields.
- `benchmark.py` for reproducible inference-latency measurement.
- Pinned dependency versions for reproducible deployment.
- `.env.example` documenting configuration boundaries.
- No secrets are stored in the repository.

## Testing
```bash
pip install -r requirements.txt
pytest -q
python benchmark.py
```

## Deployment
The Streamlit entrypoint is `app.py` in the repository root. `requirements.txt` is also in the root so Streamlit Community Cloud can install dependencies.

## Social impact / SDG
The project aligns with **UN Sustainable Development Goal 3: Good Health and Well-Being**, particularly the idea of supporting earlier awareness of non-communicable disease risk. It is a screening prototype, not a clinical diagnostic system.

## Limitations
The dataset is a standard public research dataset and is not a substitute for a representative clinical population. A production system would require external clinical validation, fairness analysis, monitoring, privacy controls, and regulatory review.

## Dataset
Pima Indians Diabetes dataset:
https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset

The application fetches the dataset with an 8-second network timeout and fails clearly if the data source is unavailable.
