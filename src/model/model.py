import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import lightning as L 
from torchmetrics import Accuracy
from torchinfo import summary

class DiabetesModel(L.LightningModule):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2 classes: 0 (non-diabetic), 1 (diabetic)
        )

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc   = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        print(f"Validation loss batch {batch_idx}: {loss.item()}")

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
