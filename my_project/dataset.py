from torchvision import datasets, transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# Data Module
class HousePricingDataModule(pl.LightningDataModule): #data loading and processing
    def __init__(self, data_dir='.', batch_size=64, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        # Download data (called only once on main process)
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Setup datasets for each stage
        if stage == 'fit' or stage is None:
            self.train_ds = datasets.MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.val_ds = datasets.MNIST(
                self.data_dir, train=False, transform=self.transform
            )
        
        if stage == 'test' or stage is None:
            self.test_ds = datasets.MNIST(
                self.data_dir, train=False, transform=self.transform
            )

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