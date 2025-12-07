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
    3. Shuffle and split into training and validation sets
    4. Replace zero values with training medians
    5. Encode categorical variables
    6. Scale numeric features manually (mean=0, std=1)
    7. Convert to PyTorch tensors and create DataLoaders
    """

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    first_dataset_path  = os.path.join(base_dir, "data", "raw", "diabetes.csv")
    second_dataset_path = os.path.join(base_dir, "data", "raw", "diabetes_prediction_dataset.csv")

    first_dataset  = pd.read_csv(first_dataset_path)
    second_dataset = pd.read_csv(second_dataset_path)

    # Fill missing columns in first dataset
    for col in ["gender", "hypertension", "heart_disease", "smoking_history", "HbA1c_level"]:
        first_dataset[col] = 0 if col != "smoking_history" else "never"
    first_dataset["blood_glucose_level"] = first_dataset["Glucose"]

    # Fill missing columns in second dataset
    for col in ["Pregnancies", "BloodPressure", "SkinThickness", "Insulin", "DiabetesPedigreeFunction", "Glucose"]:
        second_dataset[col] = 0
    second_dataset["Glucose"] = second_dataset["blood_glucose_level"]

    # Combine datasets
    combined_df = pd.concat([first_dataset, second_dataset], ignore_index=True)

    # Shuffle and split
    np.random.seed(random_seed)
    indices = np.arange(len(combined_df))
    np.random.shuffle(indices)
    split_idx = int(len(combined_df) * (1 - val_ratio))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_df = combined_df.iloc[train_idx].copy()
    val_df   = combined_df.iloc[val_idx].copy()

    # Handle missing numeric values using training medians
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                       "HbA1c_level", "blood_glucose_level"]

    for col in zero_as_missing:
        median = train_df[col].replace(0, np.nan).median()
        train_df[col] = train_df[col].replace(0, np.nan).fillna(median)
        val_df[col]   = val_df[col].replace(0, np.nan).fillna(median)

    # Encode gender
    for df in [train_df, val_df]:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).fillna(0)

    # Encode smoking history and ensure all expected columns exist
    train_smoke = pd.get_dummies(train_df['smoking_history'].replace('No Info', 'never'), 
                                 prefix='smoking_history', drop_first=False)
    val_smoke   = pd.get_dummies(val_df['smoking_history'].replace('No Info', 'never'), 
                                 prefix='smoking_history', drop_first=False)

    # Ensure all three dummy columns exist even if some categories are missing
    expected_smoking_cols = ['smoking_history_never', 'smoking_history_former', 'smoking_history_current']
    train_smoke = train_smoke.reindex(columns=expected_smoking_cols, fill_value=0)
    val_smoke   = val_smoke.reindex(columns=expected_smoking_cols, fill_value=0)

    # Add to original dataframes
    train_df = pd.concat([train_df, train_smoke], axis=1)
    val_df   = pd.concat([val_df, val_smoke], axis=1)

    # Fill any remaining NaNs with 0
    train_df = train_df.fillna(0)
    val_df   = val_df.fillna(0)

    # Define feature columns
    numeric_cols = [
        "Pregnancies", "blood_glucose_level", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigreeFunction", "Age", "HbA1c_level"
    ]
    categorical_cols = ["gender", "hypertension", "heart_disease"]
    smoking_cols = expected_smoking_cols

    feature_cols = numeric_cols + categorical_cols + smoking_cols

    # Extract X and y
    X_train = train_df[feature_cols].astype(np.float32).values
    y_train = train_df["Outcome"].fillna(train_df.get("diabetes", 0)).astype(int).values

    X_val = val_df[feature_cols].astype(np.float32).values
    y_val = val_df["Outcome"].fillna(val_df.get("diabetes", 0)).astype(int).values

    # Standardize numeric features using training mean/std
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1  # prevent division by zero

    X_train_scaled = (X_train - mean) / std
    X_train_scaled = np.clip(X_train_scaled, -10, 10)

    X_val_scaled = (X_val - mean) / std
    X_val_scaled = np.clip(X_val_scaled, -10, 10)

    # Save mean and std for inference
    scaler_dir = os.path.join(base_dir, "saved_model")
    os.makedirs(scaler_dir, exist_ok=True)
    np.save(os.path.join(scaler_dir, "mean.npy"), mean)
    np.save(os.path.join(scaler_dir, "scale.npy"), std)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor   = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_val_tensor, y_val_tensor
