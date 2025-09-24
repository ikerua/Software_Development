import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger   # 👈 nuevo
from my_project.model import HousePriceRegressor

def load_xy(path_csv: str, target_col: str = "House_Price"):
    df = pd.read_csv(path_csv)
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)

def main(args):
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

    # Set up model checkpoint callback
    ckpt_cb = ModelCheckpoint(
        dirpath=models_dir,
        filename="house_price_regressor",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=False,
    )
    # Set up early stopping callback
    es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=10)

    # CSV logger
    csv_logger = CSVLogger(save_dir=logs_dir, name="house_price")

    # Initialize PyTorch Lightning trainer
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

    print(f"Logs guardados en: {csv_logger.log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    main(args)
