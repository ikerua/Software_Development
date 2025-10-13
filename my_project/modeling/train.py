import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from my_project.model import HousePriceRegressor  # Assumed import

"""
This module contains functions and classes for training machine learning
models and generating predictions.
It includes functionalities to load data, configure training,
and execute the training process using PyTorch Lightning.
"""
def load_xy(path_csv: str, target_col: str = "House_Price"):
    """
    Load features (X) and target (y) tensors from a CSV file.

    Parameters
    ----------
    path_csv : str
        Path to the CSV file containing the data. This file is expected to be 
        in the processed data format (scaled).
    target_col : str, optional
        The name of the column containing the target variable, 
        by default "House_Price".

    Returns
    -------
    tuple of torch.Tensor
        Features X of shape (n_samples, n_features) and target y of shape (n_samples,).

    Examples
    --------
    >>> X_data, y_data = load_xy('./data/processed/train.csv')
    >>> X_data.shape, y_data.shape
    (1000, 10) (1000,)
    """
    df = pd.read_csv(path_csv)
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)


def main(args):
    """
    Configure and execute the training process with PyTorch Lightning.

    Parameters
    ----------
    args : argparse.Namespace
        Container for configuration arguments including:
        
        * batch_size (int): Size of the mini-batch for training and validation.
        * num_workers (int): Number of subprocesses for data loading.
        * lr (float): Learning rate for the optimizer.
        * weight_decay (float): L2 regularization factor.
        * epochs (int): Maximum number of epochs to train.

    Examples
    --------
    From Python:
    >>> from my_project.modeling import train
    >>> import argparse
    >>> args = argparse.Namespace(batch_size=64, num_workers=4, lr=1e-3, weight_decay=0.0, epochs=20)
    >>> train.main(args)
    """
    # Get project root directory
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_proc = os.path.join(root, "data", "processed")
    models_dir = os.path.join(root, "models")
    logs_dir = os.path.join(root, "reports", "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Load training and validation data
    X_train, y_train = load_xy(os.path.join(data_proc, "train.csv"))
    X_val, y_val = load_xy(os.path.join(data_proc, "val.csv"))

    input_dim = X_train.shape[1]

    # Create TensorDatasets for train and validation
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize model
    model = HousePriceRegressor(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Set up callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=models_dir,
        filename="house_price_regressor",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=False,
    )
    es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=10)

    # CSV logger
    csv_logger = CSVLogger(save_dir=logs_dir, name="house_price")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        deterministic=True,
        callbacks=[ckpt_cb, es_cb],
        log_every_n_steps=10,
        logger=csv_logger,
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"Logs saved at: {csv_logger.log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the House Price Regressor model.")
    parser.add_argument("--batch_size", type=int, default=64, help="Input batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 regularization term.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of training epochs.")
    args = parser.parse_args()
    main(args)
