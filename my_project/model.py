from typing import List, Optional
import torch
from torch import nn
import pytorch_lightning as pl

# Lightning Module for House Price Regression
class HousePriceRegressor(pl.LightningModule):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for easy access

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers: List[nn.Module] = []
        prev = input_dim
        # Build the neural network layers
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]  # Linear layer followed by ReLU
            prev = h
        layers += [nn.Linear(prev, 1)]  # Output layer

        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()  # Mean Squared Error loss for regression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the network
        return self.net(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        # Training step: compute predictions and loss
        x, y = batch
        y_hat = self(x.float())
        loss = self.loss_fn(y_hat, y.float())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step: compute predictions and loss
        x, y = batch
        y_hat = self(x.float())
        loss = self.loss_fn(y_hat, y.float())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # Test step: compute predictions and loss
        x, y = batch
        y_hat = self(x.float())
        loss = self.loss_fn(y_hat, y.float())
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        # Configure optimizer (Adam) for training
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
