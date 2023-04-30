import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .base_dataset import PredictDataset, TrainDataset


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=None,
        shuffle=False,
        num_workers=1,
        prepare_data_per_node=False,
        preprocess_fn=None,
        kfold=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prepare_data_per_node = prepare_data_per_node

    def prepare_data(self):
        train_path = self.data_dir + "/train.csv"
        self.train_df = pd.read_csv(train_path)

        predict_path = self.data_dir + "/predict.csv"
        self.predict_df = pd.read_csv(predict_path)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = TrainDataset(self.train_df)
            
        elif stage == "predict":
            self.predict_dataset = PredictDataset(self.predict_df)

    def train_dataloader(self):
        # Implement your train dataloader logic here
        raise NotImplementedError

    def val_dataloader(self):
        # Implement your validation dataloader logic here
        raise NotImplementedError

    def test_dataloader(self):
        # Implement your test dataloader logic here
        raise NotImplementedError

    def predict_dataloader(self):
        if self.predict_dataset is None:
            raise ValueError("Predict dataset is not provided.")
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    # Optional methods
    def transfer_batch_to_device(self, batch, device):
        return super().transfer_batch_to_device(batch, device)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return super().on_before_batch_transfer(batch, dataloader_idx)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return super().on_after_batch_transfer(batch, dataloader_idx)

    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict)

    def state_dict(self):
        return super().state_dict()

    def teardown(self, stage=None):
        super().teardown(stage)
