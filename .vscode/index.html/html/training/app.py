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


if __name__ == "__main__":
    app.run(debug=True)


