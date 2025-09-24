import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from joblib import dump
import os
# Data Module
class HousePricingDataModule(pl.LightningDataModule): #data loading and processing
    def __init__(self, data_dir='.', batch_size=64, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        

        data_interim_dir = '../data/interim/'
        data_processed_dir = '../data/processed/'
        os.makedirs(data_interim_dir, exist_ok=True)
        os.makedirs(data_processed_dir, exist_ok=True)

        # 1) Cargar dataset crudo
        df = pd.read_csv(self.data_dir)
        print(f"Dataset loaded with shape: {df.shape}")

        X, y = df.drop('House_Price', axis=1), df['House_Price']
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")

        # 2) Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        # Guardar INTERIM (sin escalar)
        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(data_interim_dir, 'train.csv'), index=False)
        pd.concat([X_val,   y_val],   axis=1).to_csv(os.path.join(data_interim_dir, 'val.csv'),   index=False)
        pd.concat([X_test,  y_test],  axis=1).to_csv(os.path.join(data_interim_dir, 'test.csv'),  index=False)
        print("Interim files saved.")

        # 3) Escalado X & y
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_s = pd.DataFrame(x_scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_s   = pd.DataFrame(x_scaler.transform(X_val),       columns=X_val.columns)
        X_test_s  = pd.DataFrame(x_scaler.transform(X_test),      columns=X_test.columns)

        y_train_s = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()
        y_val_s   = y_scaler.transform(y_val.to_numpy().reshape(-1, 1)).ravel()
        y_test_s  = y_scaler.transform(y_test.to_numpy().reshape(-1, 1)).ravel()

        # 4) Guardar PROCESSED (y escalado)
        X_train_s['House_Price'] = y_train_s
        X_val_s['House_Price']   = y_val_s
        X_test_s['House_Price']  = y_test_s

        X_train_s.to_csv(os.path.join(data_processed_dir, 'train.csv'), index=False)
        X_val_s.to_csv(os.path.join(data_processed_dir, 'val.csv'),   index=False)
        X_test_s.to_csv(os.path.join(data_processed_dir, 'test.csv'),  index=False)
        print("Processed files saved (scaled X & y).")

        # 5) Guardar scalers
        dump(x_scaler, os.path.join(data_processed_dir, 'x_scaler.joblib'))
        dump(y_scaler, os.path.join(data_processed_dir, 'y_scaler.joblib'))
        print("Scalers saved: x_scaler.joblib, y_scaler.joblib")


    def setup(self, stage=None):
        data_processed_dir = '../data/processed/'
        # Setup datasets for each stage
        if stage == 'fit' or stage is None:
            self.train_ds = pd.read_csv(data_processed_dir + 'train.csv')
            self.val_ds = pd.read_csv(data_processed_dir + 'val.csv')

        if stage == 'test' or stage is None:
            self.test_ds = pd.read_csv(data_processed_dir + 'test.csv')

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=1000, 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=1000, 
            num_workers=self.num_workers
        )