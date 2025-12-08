import torch
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

from src.model.model_dgg import DiabetesModel

# Load scaler values
mean = np.load("saved_model/mean_dgg.npy")
scale = np.load("saved_model/scale_dgg.npy")

# Load model
input_size = len(mean)  # 15 features
model = DiabetesModel(input_size=input_size)
model.load_state_dict(torch.load("saved_model/model_dgg.pt", map_location=torch.device("cpu")))
model.eval()

# Define feature order exactly as in training
FEATURE_ORDER = [
    "Pregnancies", "blood_glucose_level", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "HbA1c_level",
    "gender", "hypertension", "heart_disease",
    "smoking_history_never", "smoking_history_former", "smoking_history_current"
]

# Map smoking input to one-hot encoding
SMOKING_MAP = {
    "never": [1, 0, 0],
    "former": [0, 1, 0],
    "current": [0, 0, 1]
}

def prepare_input(user_input):
    """Prepare input array in the correct order for the model."""
    x = np.zeros(len(FEATURE_ORDER), dtype=float)

    for i, feat in enumerate(FEATURE_ORDER):
        if feat.startswith("smoking_history_"):
            # Handle smoking one-hot
            smoking = user_input.get("smoking", "never")
            x[i:i+3] = SMOKING_MAP.get(smoking, [1,0,0])
            break  # all three are filled at once
        elif feat == "gender":
            gender = user_input.get("gender", "Male")
            x[i] = 1.0 if gender == "Male" else 0.0
        else:
            x[i] = float(user_input.get(feat, 0))
    return x

def predict_diabetes(user_input):
    """
    Predict diabetes from a user input dictionary.
    Returns:
        pred_class: 0 (non-diabetic) or 1 (diabetic)
        pred_prob: probability of predicted class
    """
    x = prepare_input(user_input)
    x_scaled = (x - mean) / scale
    x_tensor = torch.tensor([x_scaled], dtype=torch.float32)

    logits = model(x_tensor)
    pred_class = torch.argmax(logits, dim=1).item()
    pred_prob = torch.softmax(logits, dim=1)[0, pred_class].item()

    return pred_class, pred_prob
