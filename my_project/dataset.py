"""
This module contains the DataModule for the House Price Regression task.
It handles loading, splitting, scaling, and creating data loaders for
the dataset.
"""

__docformat__ = "numpy"

import importlib.resources
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from joblib import dump
import os
import torch
from typing import Optional

# Data Module

class HousePricingDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule handling the full data pipeline for the
    House Price Regression task, including loading, splitting, scaling,
    and creating data loaders.

    Attributes
    ----------
    data_dir : str
        Path to the raw input CSV file.
    batch_size : int
        Batch size for the training DataLoader.
    num_workers : int
        Number of workers for data loading.
    train_ds : pandas.DataFrame or None
        Training dataset containing scaled features and target.
    val_ds : pandas.DataFrame or None
        Validation dataset containing scaled features and target.
    test_ds : pandas.DataFrame or None
        Test dataset containing scaled features and target.

    Examples
    --------
    >>> data_module = HousePricingDataModule(data_dir='./data.csv', batch_size=32)
    >>> data_module.prepare_data()
    >>> data_module.setup(stage='fit')
    >>> train_loader = data_module.train_dataloader()
    """

    def __init__(self, data_dir='.', train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, batch_size=64, num_workers=8):
        """
        Initializes the data module settings.

        Parameters
        ----------
        data_dir : str, optional
            Path to the raw CSV file containing the data. 
            By default '.', which should be replaced by the actual path to the raw data file.
        batch_size : int, optional
            The size of the mini-batch used during training, by default 64.
        num_workers : int, optional
            The number of subprocesses to use for data loading, by default 8.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None


    def prepare_data(self):
        """
        Preprocess the dataset and save interim and processed files.
        This method is designed to be run only once.
        """
        data_pkg_path = importlib.resources.files("data")
    
        data_processed_dir = data_pkg_path / "processed"
        os.makedirs(data_processed_dir, exist_ok=True)

        # Si los datos ya están procesados, no hacer nada más.
        if os.path.exists(os.path.join(data_processed_dir, 'train.csv')):
            print("Data already prepared. Skipping preparation step.")
            return

        # 1) Cargar dataset crudo
        df = pd.read_csv(self.data_dir)
        print(f"Raw dataset loaded with shape: {df.shape}")

        X, y = df.drop('House_Price', axis=1), df['House_Price']
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")

        # 2) Split: 60% Train, 20% Val, 20% Test
        # Primero, separamos el conjunto de test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_ratio, random_state=42
        )
        
        # Luego, separamos el resto en train y validation
        # El ratio de validación se ajusta al tamaño restante
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=42
        )
        
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

        # 3) Escalado X & y
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_s = pd.DataFrame(x_scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_s   = pd.DataFrame(x_scaler.transform(X_val), columns=X_val.columns)
        X_test_s  = pd.DataFrame(x_scaler.transform(X_test), columns=X_test.columns)

        y_train_s = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()
        y_val_s   = y_scaler.transform(y_val.to_numpy().reshape(-1, 1)).ravel()
        y_test_s  = y_scaler.transform(y_test.to_numpy().reshape(-1, 1)).ravel()

        # 4) Guardar PROCESSED (con datos escalados)
        X_train_s['House_Price'] = y_train_s
        X_val_s['House_Price']   = y_val_s
        X_test_s['House_Price']  = y_test_s

        X_train_s.to_csv(os.path.join(data_processed_dir, 'train.csv'), index=False)
        X_val_s.to_csv(os.path.join(data_processed_dir, 'val.csv'), index=False)
        X_test_s.to_csv(os.path.join(data_processed_dir, 'test.csv'), index=False)
        print("Processed files saved (scaled X & y).")

        # 5) Guardar scalers
        dump(x_scaler, os.path.join(data_processed_dir, 'x_scaler.joblib'))
        dump(y_scaler, os.path.join(data_processed_dir, 'y_scaler.joblib'))
        print("Scalers saved: x_scaler.joblib, y_scaler.joblib")
       


    def setup(self, stage: Optional[str] = None):
        """
        Loads the pre-processed datasets from the processed data
        directory into pandas DataFrames.

        Parameters
        ----------
        stage : str or None, optional
            The current stage of training ('fit', 'validate', 'test', 'predict'). 
            If 'fit' or None, loads train and validation sets. If 'test' or None, 
            loads the test set. By default None.
        """
        data_processed_dir = importlib.resources.files("data") / "processed"
        # Setup datasets for each stage
        if stage == 'fit' or stage is None:
            self.train_ds = pd.read_csv(data_processed_dir / 'train.csv')
            self.val_ds = pd.read_csv(data_processed_dir / 'val.csv')

        if stage == 'test' or stage is None:
            self.test_ds = pd.read_csv(data_processed_dir / 'test.csv')

    def train_dataloader(self):
        """
        Returns the DataLoader for the training set.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader configured with `self.batch_size`, shuffling enabled, 
            and `self.num_workers`.
        """
        # We need to explicitly convert the DataFrame to Tensors for the DataLoader
        train_features = torch.tensor(self.train_ds.drop('House_Price', axis=1).values, dtype=torch.float32)
        train_targets = torch.tensor(self.train_ds['House_Price'].values, dtype=torch.float32)
        train_dataset = torch.utils.data.TensorDataset(train_features, train_targets)
        
        return DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation set.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader configured with a fixed batch size of 1000 and 
            no shuffling.
        """
        val_features = torch.tensor(self.val_ds.drop('House_Price', axis=1).values, dtype=torch.float32)
        val_targets = torch.tensor(self.val_ds['House_Price'].values, dtype=torch.float32)
        val_dataset = torch.utils.data.TensorDataset(val_features, val_targets)

        return DataLoader(
            val_dataset, 
            batch_size=1000, 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        """
        Returns the DataLoader for the test set.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader configured with a fixed batch size of 1000 and 
            no shuffling.
        """
        test_features = torch.tensor(self.test_ds.drop('House_Price', axis=1).values, dtype=torch.float32)
        test_targets = torch.tensor(self.test_ds['House_Price'].values, dtype=torch.float32)
        test_dataset = torch.utils.data.TensorDataset(test_features, test_targets)
        
        return DataLoader(
            test_dataset, 
            batch_size=1000, 
            num_workers=self.num_workers
        )