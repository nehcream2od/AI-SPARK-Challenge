from abc import abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch.nn as nn


class BaseLitModel(pl.LightningModule):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def training_step(self, batch, batch_idx):
        # Implement your training loop logic here
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        # Implement your validation loop logic here
        raise NotImplementedError

    def configure_optimizers(self):
        # Implement your optimizer logic here
        raise NotImplementedError
