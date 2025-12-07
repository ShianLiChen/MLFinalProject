import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(batch_size=32, val_ratio=0.2, random_seed=42):
    """
    Load and preprocess the combined Diabetes dataset

    Steps:
    1. Load first and second datasets
    2. Align columns and fill missing values
    3. Encode categorical variables
    4. Scale numeric features manually (mean=0, std=1)
    5. Split into training and validation sets
    6. Convert to PyTorch tensors and return DataLoaders
    """

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    first_dataset_path  = os.path.join(base_dir, "data", "raw", "diabetes.csv")
    second_dataset_path = os.path.join(base_dir, "data", "raw", "diabetes_prediction_dataset.csv")

    first_dataset  = pd.read_csv(first_dataset_path)
    second_dataset = pd.read_csv(second_dataset_path)

    # Fill missing columns in the first dataset
    for col in ["gender", "hypertension", "heart_disease", "smoking_history", "HbA1c_level"]:
        if col != "smoking_history":
            first_dataset[col] = 0
        else:
            first_dataset[col] = "never"
    first_dataset["blood_glucose_level"] = first_dataset["Glucose"]

    # Fill missing columns in the second dataset
    for col in ["Pregnancies", "BloodPressure", "SkinThickness", "Insulin", "DiabetesPedigreeFunction", "Glucose"]:
        second_dataset[col] = 0
    second_dataset["Glucose"] = second_dataset["blood_glucose_level"]

    # Combine datasets
    combined_df = pd.concat([first_dataset, second_dataset], ignore_index=True)

    # Handle missing numeric values
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                       "HbA1c_level", "blood_glucose_level"]
    for col in zero_as_missing:
        combined_df[col] = combined_df[col].replace(0, np.nan)
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())

    # Encode gender
    combined_df['gender'] = combined_df['gender'].map({'Male': 1, 'Female': 0})
    combined_df['gender'] = combined_df['gender'].fillna(0)

    # Encode smoking history
    combined_df['smoking_history'] = combined_df['smoking_history'].replace('No Info', 'never')
    combined_df = pd.get_dummies(combined_df, columns=['smoking_history'], drop_first=False)

    # Fill any remaining NaNs
    combined_df = combined_df.fillna(0)

    # Define numeric and categorical columns
    numeric_cols = [
        "Pregnancies", "blood_glucose_level", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigreeFunction", "Age", "HbA1c_level"
    ]
    categorical_cols = ["gender", "hypertension", "heart_disease"]
    smoking_cols = ["smoking_history_never", "smoking_history_former", "smoking_history_current"]

    feature_cols = numeric_cols + categorical_cols + smoking_cols
    X = combined_df[feature_cols].astype(np.float32).values  # convert to NumPy array
    y = combined_df["Outcome"].fillna(combined_df["diabetes"]).astype(int).values

    # Manual standardization
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0] = 1  # prevent division by zero
    X_scaled = (X - mean) / std
    X_scaled = np.clip(X_scaled, -10, 10)  # clip extreme values

    # Save mean and std for inference
    scaler_dir = os.path.join(base_dir, "saved_model")
    os.makedirs(scaler_dir, exist_ok=True)
    np.save(os.path.join(scaler_dir, "mean.npy"), mean)
    np.save(os.path.join(scaler_dir, "scale.npy"), std)

    # Shuffle and split
    np.random.seed(random_seed)
    indices = np.arange(len(X_scaled))
    np.random.shuffle(indices)
    split_idx = int(len(X_scaled) * (1 - val_ratio))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = X_scaled[train_idx], y[train_idx]
    X_val, y_val     = X_scaled[val_idx], y[val_idx]

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    y_val   = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_val, y_val
