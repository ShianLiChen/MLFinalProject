from flask import Flask, render_template, request
from model_utils import predict_diabetes
import numpy as np
import os

app = Flask(__name__)

# Load median or mean values for default inputs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_DIR = os.path.join(BASE_DIR, "..", "saved_model")

# Load means
mean = np.load(os.path.join(SCALER_DIR, "mean.npy"))
# Map features in the same order as your model expects
FEATURE_ORDER = [
    "Pregnancies", "blood_glucose_level", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "HbA1c_level",
    "gender", "hypertension", "heart_disease",
    "smoking_history_never", "smoking_history_former", "smoking_history_current"
]

# Create a dict of defaults for numeric fields
NUMERIC_DEFAULTS = dict(zip(FEATURE_ORDER[:9], mean[:9]))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prob = None
    if request.method == "POST":
        user_input = {
            "Pregnancies": request.form.get("Pregnancies", NUMERIC_DEFAULTS["Pregnancies"]),
            "blood_glucose_level": request.form.get("blood_glucose_level", NUMERIC_DEFAULTS["blood_glucose_level"]),
            "BloodPressure": request.form.get("BloodPressure", NUMERIC_DEFAULTS["BloodPressure"]),
            "SkinThickness": request.form.get("SkinThickness", NUMERIC_DEFAULTS["SkinThickness"]),
            "Insulin": request.form.get("Insulin", NUMERIC_DEFAULTS["Insulin"]),
            "BMI": request.form.get("BMI", NUMERIC_DEFAULTS["BMI"]),
            "DiabetesPedigreeFunction": request.form.get("DiabetesPedigreeFunction", NUMERIC_DEFAULTS["DiabetesPedigreeFunction"]),
            "Age": request.form.get("Age", NUMERIC_DEFAULTS["Age"]),
            "HbA1c_level": request.form.get("HbA1c_level", NUMERIC_DEFAULTS["HbA1c_level"]),
            "gender": request.form.get("gender", "Male"),
            "hypertension": request.form.get("hypertension", 0),
            "heart_disease": request.form.get("heart_disease", 0),
            "smoking": request.form.get("smoking", "never")
        }

        # Convert numeric fields to float
        numeric_fields = [
            "Pregnancies", "blood_glucose_level", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "HbA1c_level",
            "hypertension", "heart_disease"
        ]
        for field in numeric_fields:
            user_input[field] = float(user_input.get(field, NUMERIC_DEFAULTS.get(field, 0)))

        result, prob = predict_diabetes(user_input)

    return render_template("index.html", result=result, prob=prob, defaults=NUMERIC_DEFAULTS)

if __name__ == "__main__":
    app.run(debug=True)
