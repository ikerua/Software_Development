from typing import List, Optional
import torch
from torch import nn
import pytorch_lightning as pl


class HousePriceRegressor(pl.LightningModule):
    """
    PyTorch Lightning module for house price regression.

    Implements a simple feed-forward neural network (MLP) trained with
    Mean Squared Error loss and optimized using Adam.

    Attributes
    ----------
    net : torch.nn.Sequential
        Neural network layers (MLP).
    loss_fn : torch.nn.MSELoss
        Mean Squared Error loss function.

    Examples
    --------
    >>> model = HousePriceRegressor(input_dim=5)
    >>> x = torch.randn(2, 5)
    >>> preds = model(x)
    >>> preds.shape
    torch.Size([2])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        """
        Initialize model architecture and hyperparameters.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dims : list of int, optional
            Sizes of hidden layers. Defaults to [64, 32].
        lr : float, optional
            Learning rate for Adam optimizer, by default 1e-3.
        weight_decay : float, optional
            L2 regularization term, by default 0.0.
        """
        super().__init__()
        self.save_hyperparameters()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]  # Output layer

        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch_size,).

        Examples
        --------
        >>> model = HousePriceRegressor(input_dim=5)
        >>> x = torch.randn(3, 5)
        >>> model(x).shape
        torch.Size([3])
        """
        return self.net(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step and log the loss.

        Parameters
        ----------
        batch : tuple
            (features, targets).
        batch_idx : int
            Index of current batch.

        Returns
        -------
        torch.Tensor
            Training loss (MSE).
        """
        x, y = batch
        y_hat = self(x.float())
        loss = self.loss_fn(y_hat, y.float())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step and log the loss.

        Parameters
        ----------
        batch : tuple
            (features, targets).
        batch_idx : int
            Index of current batch.

        Returns
        -------
        None
        """
        x, y = batch
        y_hat = self(x.float())
        loss = self.loss_fn(y_hat, y.float())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step and log the loss.

        Parameters
        ----------
        batch : tuple
            (features, targets).
        batch_idx : int
            Index of current batch.

        Returns
        -------
        None
        """
        x, y = batch
        y_hat = self(x.float())
        loss = self.loss_fn(y_hat, y.float())
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """
        Set up the optimizer (Adam).

        Returns
        -------
        torch.optim.Adam
            Optimizer configured with model hyperparameters.

        Examples
        --------
        >>> model = HousePriceRegressor(input_dim=10)
        >>> opt = model.configure_optimizers()
        >>> type(opt)
        <class 'torch.optim.adam.Adam'>
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
