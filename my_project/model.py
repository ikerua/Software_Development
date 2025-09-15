import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

# Model (unchanged) 
class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(16*13*13, 32)
        self.fc2 = nn.Linear(32, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = self.flat(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log('test_acc', acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters())
