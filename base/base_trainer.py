import pytorch_lightning as pl
import torch


class BaseLitModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Implement your training loop logic here
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        # Implement your validation loop logic here
        raise NotImplementedError

    def configure_optimizers(self):
        # Implement your optimizer logic here
        raise NotImplementedError
