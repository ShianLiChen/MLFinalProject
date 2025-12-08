import pandas as pd

import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer

import src.model.model_idg as model

import os
from lightning.pytorch.callbacks import EarlyStopping

def main():

    print("Cuda avail: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Cuda device: ", torch.cuda.get_device_name(0))

    # Base directory of the repo
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    # Paths for input dataset and model output
    input_csv = os.path.join(BASE_DIR, "data", "raw", "home_insulin_clean_target_gc.csv")
    model_path = os.path.join(BASE_DIR, "saved_model", "model_idg.pt")

    # Load cleaned dataset from data/raw
    df: pd.DataFrame = pd.read_csv(input_csv, parse_dates=["dateTime"])

    # Split dataset 50/50 for training/testing
    split_idx = int(len(df) * 0.5)

    train_df = model.InsulinDataset(df.iloc[:split_idx])
    test_df  = model.InsulinDataset(df.iloc[split_idx:])

    train_loader = DataLoader(
        train_df, 
        batch_size=32, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_df, 
        batch_size=32, 
        shuffle=False
    )

    # Initialize model
    uma = model.InsulinModule()

    # Train the model
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

    trainer = Trainer(
        max_epochs=75,
        deterministic=True,
        callbacks=[early_stop]
    )
    trainer.fit(uma,
                train_dataloaders=train_loader,
                val_dataloaders=test_loader)

    # Save final model
    torch.save(uma.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
