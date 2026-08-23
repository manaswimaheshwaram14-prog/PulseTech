from flask import Flask, render_template, request, jsonify,redirect
import pickle
import pandas as pd
import warnings
import json
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import joblib

app = Flask(__name__, template_folder=".")
model = joblib.load("pulsetech_sepsis_model.pkl")

# Authorized users
# Load registered users
def load_users():
    try:
        with open("users.json", "r") as file:
            return json.load(file)
    except:
        return {}

USERS = load_users()


@app.route("/")
def login():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login_user():
    username = request.form.get("username")
    password = request.form.get("password")

    if username not in USERS:
        return render_template(
            "login.html",
            error="Account not found. Please Sign Up first."
        )

    if USERS[username] != password:
        return render_template(
            "login.html",
            error="Incorrect password."
        )

    return render_template("dashboard.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")

    username = request.form.get("username")
    password = request.form.get("password")

    # Check if username already exists
    if username in USERS:
        return render_template(
            "signup.html",
            error="Username already exists"
        )

    # Add new user
    USERS[username] = password

    # Save users to users.json
    with open("users.json", "w") as file:
        json.dump(USERS, file, indent=4)

    return render_template(
        "login.html",
        success="Account created successfully! Please login."
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    heart_rate = float(data["heart_rate"])
    systolic_bp = float(data["systolic_bp"])
    spo2 = float(data["spo2"])
    lactate = float(data["lactate"])

    features = pd.DataFrame([{
    "heart_rate": heart_rate,
    "systolic_bp": systolic_bp,
    "spo2": spo2,
    "lactate": lactate
}])

    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features)[0][1]
    else:
        probability = 0.0

    return jsonify({
        "prediction": int(prediction),
        "probability": round(float(probability) * 100, 2)
    })
@app.route("/logout")
def logout():
    return redirect("/")

PATIENTS = {
    "04": {
        "bed": "04",
        "name": "Arthur Dent",
        "age": "68M",
        "condition": "Post-Op Peritonitis",
        "risk": 88,
        "risk_level": "HIGH",
        "heart_rate": [88, 92, 96, 101, 106, 109, 112],
        "bp_systolic": [110, 106, 102, 98, 94, 91, 88],
        "bp_diastolic": [70, 68, 66, 63, 60, 57, 54],
        "spo2": [98, 97, 96, 95, 94, 92, 91],
        "lactate": [1.8, 2.0, 2.2, 2.5, 2.8, 3.3, 3.8]
    },

    "07": {
        "bed": "07",
        "name": "Elena Rostova",
        "age": "54F",
        "condition": "Acute Pyelonephritis",
        "risk": 54,
        "risk_level": "MEDIUM",
        "heart_rate": [82, 84, 86, 89, 91, 94, 96],
        "bp_systolic": [118, 116, 114, 112, 110, 108, 106],
        "bp_diastolic": [76, 75, 74, 72, 70, 69, 68],
        "spo2": [98, 98, 97, 97, 96, 96, 95],
        "lactate": [1.4, 1.5, 1.7, 1.8, 2.0, 2.1, 2.3]
    },

    "12": {
        "bed": "12",
        "name": "Marcus Vance",
        "age": "42M",
        "condition": "Observation",
        "risk": 12,
        "risk_level": "LOW",
        "heart_rate": [72, 73, 72, 74, 73, 72, 71],
        "bp_systolic": [122, 121, 123, 122, 124, 123, 122],
        "bp_diastolic": [80, 79, 80, 81, 80, 79, 80],
        "spo2": [99, 99, 98, 99, 99, 98, 99],
        "lactate": [1.1, 1.0, 1.1, 1.0, 1.1, 1.0, 1.0]
    }
}


@app.route("/patient-details/<bed>")
def patient_details(bed):
    bed = bed.strip()

    if bed not in PATIENTS:
        return "Patient not found", 404

    patient = PATIENTS[bed]

    return render_template(
        "patient.html",
        patient=patient
    )


@app.route("/recommendation/<bed>")
def recommendation(bed):
    bed = bed.strip()

    if bed not in PATIENTS:
        return "Patient not found", 404

    patient = PATIENTS[bed]

    return render_template(
        "recommendation.html",
        patient=patient
    )
@app.route("/patient/<bed>")
def get_patient(bed):
    bed = bed.strip()

    if bed not in PATIENTS:
        return jsonify({
            "error": f"Bed {bed} not found."
        }), 404

    return jsonify(PATIENTS[bed])
if __name__ == "__main__":
    app.run(debug=True)


