import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from src.data.clean_data_dgg import load_data
from src.model.model_dgg import DiabetesModel
from torchinfo import summary
import os

def main():
    # Load data
    train_loader, val_loader, X_val, y_val = load_data(batch_size=32)

    # Determine input size dynamically
    input_size = X_val.shape[1]

    # Initialize model
    model = DiabetesModel(input_size=input_size)

    # Print model summary
    summary(model, input_size=(1, input_size))

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # metric to monitor
        patience=5,          # stop if no improvement for 5 epochs
        mode="min"
    )

    # Trainer with early stopping
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        callbacks=[early_stop_callback]  # add the callback here
    )


    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Save the trained model
    saved_model_dir = os.path.join("saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(saved_model_dir, "model_dgg.pt"))

    # Evaluate final accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_val).float().mean().item()
    print(f"Final Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
