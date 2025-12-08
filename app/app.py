from flask import Flask, render_template, request
from model_utils import predict_diabetes
import numpy as np
import os
from src.model.model_idg import InsulinModule, InsulinDataset
import torch

app = Flask(__name__)

# Load median or mean values for default inputs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_DIR = os.path.join(BASE_DIR, "..", "saved_model")

# Load means
mean = np.load(os.path.join(SCALER_DIR, "mean_dgg.npy"))
# Map features in the same order as your model expects
FEATURE_ORDER = [
    "Pregnancies", "blood_glucose_level", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "HbA1c_level",
    "gender", "hypertension", "heart_disease",
    "smoking_history_never", "smoking_history_former", "smoking_history_current"
]

# Create a dict of defaults for numeric fields
NUMERIC_DEFAULTS = dict(zip(FEATURE_ORDER[:9], mean[:9]))

# Setup insulin predictor model
INSULIN_MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_model", "model_idg.pt")
insulin_model = InsulinModule()
insulin_model.load_state_dict(torch.load(INSULIN_MODEL_PATH, map_location="cpu"))
insulin_model.eval()

INSULIN_DEFAULTS = {
    "netCarbs": 50.0,
    "bloodGlucose": 120.0,
    "targetGlucose": 100.0,
    "hour": 12
}

def preprocess_insulin_input(netCarbs, bloodGlucose, targetGlucose, hour):
    """Preprocess input for InsulinModule."""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    batch_dict = {
        "hour_sin": torch.tensor([[hour_sin]], dtype=torch.float32),
        "hour_cos": torch.tensor([[hour_cos]], dtype=torch.float32),
        "netCarbs": torch.tensor([[netCarbs]], dtype=torch.float32),
        "bloodGlucose": torch.tensor([[bloodGlucose]], dtype=torch.float32),
        "targetGlucose": torch.tensor([[targetGlucose]], dtype=torch.float32)
    }
    return batch_dict


@app.route("/", methods=["GET", "POST"])
def index():
    """Route for loading the diabetic diagnosis prediction page"""
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

@app.route("/insulin", methods=["GET", "POST"])
def insulin_predictor():
    """Route for loading the insulin dosage predictor"""
    result = None
    if request.method == "POST":
        netCarbs = float(request.form.get("netCarbs", INSULIN_DEFAULTS["netCarbs"]))
        bloodGlucose = float(request.form.get("bloodGlucose", INSULIN_DEFAULTS["bloodGlucose"]))
        targetGlucose = float(request.form.get("targetGlucose", INSULIN_DEFAULTS["targetGlucose"]))
        hour = int(request.form.get("hour", INSULIN_DEFAULTS["hour"]))

        batch_dict = preprocess_insulin_input(netCarbs, bloodGlucose, targetGlucose, hour)
        with torch.no_grad():
            result = insulin_model(batch_dict).item()

    return render_template("insulin_predictor.html", result=result, defaults=INSULIN_DEFAULTS)

if __name__ == "__main__":
    app.run(debug=True)
