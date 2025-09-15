import model
data_module = MNISTDataModule()
net = Net()
trainer = pl.Trainer(max_epochs=3, accelerator="auto", devices="auto")
trainer.fit(net, datamodule=data_module)
trainer.test(net, datamodule=data_module)
