import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
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
        # Load from CSV data/raw/house_price_regression_dataset.csv
        df = pd.read_csv(self.data_dir)
    
        print(f"Dataset loaded with shape: {df.shape}")

        # Basic data splitting
        X,y = df.drop('House_Price', axis=1), df['House_Price']
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")

        # Split the data into train, val, test sets
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.25,random_state=42) 
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # Save interim data
        train_df.to_csv(data_interim_dir + 'train.csv', index=False)
        val_df.to_csv(data_interim_dir + 'val.csv', index=False)
        test_df.to_csv(data_interim_dir + 'test.csv', index=False)
        print("Data split into train, val, and test sets. Interim files saved.")

        # Basic data processing (e.g., normalization)
        normalizer = StandardScaler()
        X_train_norm = pd.DataFrame(normalizer.fit_transform(X_train), columns=X_train.columns)
        X_train_norm['House_Price'] = y_train.values
        X_val_norm = pd.DataFrame(normalizer.transform(X_val), columns=X_val.columns)
        X_val_norm['House_Price'] = y_val.values
        X_test_norm = pd.DataFrame(normalizer.transform(X_test), columns=X_test.columns)
        X_test_norm['House_Price'] = y_test.values

        # Save processed data
        X_train_norm.to_csv(data_processed_dir + 'train.csv', index=False)
        X_val_norm.to_csv(data_processed_dir + 'val.csv', index=False)
        X_test_norm.to_csv(data_processed_dir + 'test.csv', index=False)
        print("Data normalized and processed files saved.")
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