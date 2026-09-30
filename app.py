import math

print("=======================================================")
print("  Early Diabetes Risk Clinical Inference Engine  ")
print("=======================================================\n")

# Model parameters (Pre-trained Logistic Regression weights)
# Features: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Pedigree, Age]
weights = [0.1232, 0.0352, -0.0133, 0.0006, -0.0012, 0.0897, 0.9452, 0.0149]
intercept = -8.4047

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def predict_risk(features):
    logit = intercept + sum(w * x for w, x in zip(weights, features))
    return sigmoid(logit)

# Test Patient Clinical Data
# Format: [Pregnancies, Glucose, BP, Skin Thickness, Insulin, BMI, Pedigree, Age]
sample_patient = [2, 145.0, 78.0, 25.0, 95.0, 29.8, 0.52, 42]

print("Patient Health Indicators:")
print(f"- Age: {int(sample_patient[7])} years")
print(f"- Glucose Level: {sample_patient[1]} mg/dL")
print(f"- Blood Pressure: {sample_patient[2]} mm Hg")
print(f"- Body Mass Index (BMI): {sample_patient[5]}")
print(f"- Insulin: {sample_patient[4]} mu U/ml")

risk_score = predict_risk(sample_patient)
risk_percentage = risk_score * 100.0

print("\n" + "-" * 40)
print(f"Calculated Diabetes Risk: {risk_percentage:.2f}%")

if risk_score >= 0.50:
    print("Classification Result: [HIGH RISK]")
    print("Clinical Note: Schedule diagnostic HbA1c test.")
else:
    print("Classification Result: [LOW RISK]")
    print("Clinical Note: Normal risk profile. Routine checkups recommended.")
print("-" * 40)
