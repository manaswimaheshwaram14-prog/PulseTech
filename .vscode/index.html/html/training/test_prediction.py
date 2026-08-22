import joblib
import pandas as pd

model = joblib.load("pulsetech_sepsis_model.pkl")

new_patient = pd.DataFrame([{
    "heart_rate": 118,
    "systolic_bp": 88,
    "spo2": 91,
    "lactate": 3.8
}])

prediction = model.predict(new_patient)[0]
probability = model.predict_proba(new_patient)[0][1] * 100

print("Prediction:", prediction)
print("Risk Probability:", round(probability, 2), "%")