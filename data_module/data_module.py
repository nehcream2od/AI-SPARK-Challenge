import numpy as np
import pandas as pd
from base import BaseDataModule
from datasets import CustomPredictDataset, CustomTrainDataset
from .scaler import GaussRankScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def apply_scaler(train, predict, scaler_type):
    if scaler_type == "standard":
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train)
        scaled_predict = scaler.transform(predict)
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train)
        scaled_predict = scaler.transform(predict)
    elif scaler_type == "robust":
        scaler = RobustScaler()
        scaled_train = scaler.fit_transform(train)
        scaled_predict = scaler.transform(predict)
    elif scaler_type == "gaussrank":
        scaler = GaussRankScaler()
        train.drop(["out_pressure"], axis=1, inplace=True)
        predict.drop(["out_pressure"], axis=1, inplace=True)
        scaled_train = scaler.fit_transform(train)
        scaled_predict = scaler.transform(predict)
    elif scaler_type == "log":
        scaled_train = np.log1p(train).values
        scaled_predict = np.log(predict).values
    else:
        raise ValueError("Unknown scaler type")

    return scaled_train, scaled_predict


def apply_fourier_transform(df, features):
    fourier_df = df.copy()
    for feature in features:
        fourier_transformed_feature = np.fft.fft(df[feature])
        fourier_df[f"{feature}_fft_real"] = fourier_transformed_feature.real
        fourier_df[f"{feature}_fft_imag"] = fourier_transformed_feature.imag
    return fourier_df


class CustomDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle,
        num_workers,
        prepare_data_per_node,
        kfold=None,
        preprocess_fn=None,
        **kwargs,
    ):
        super().__init__(data_dir, **kwargs)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prepare_data_per_node = prepare_data_per_node
        self.preprocess_fn = preprocess_fn
        self.kfold = kfold
        self.save_hyperparameters()

    def prepare_data(self):
        self.train_df = pd.read_csv(self.data_dir + "/train.csv")
        self.predict_df = pd.read_csv(self.data_dir + "/predict.csv")

    def setup(self, stage=None):
        self.train_datasets = []
        self.predict_datasets = []

        for tp in self.train_df["type"].unique():
            train_subset = self.train_df[self.train_df["type"] == tp].drop(
                "type", axis=1
            )
            predict_subset = self.predict_df[self.predict_df["type"] == tp].drop(
                "type", axis=1
            )

            if self.preprocess_fn:
                train_subset, predict_subset = self._apply_preprocessing(
                    train_subset, predict_subset
                )

            self.train_datasets.append(CustomTrainDataset(train_subset))
            self.predict_datasets.append(CustomPredictDataset(predict_subset))

    def _apply_preprocessing(self, train_subset, predict_subset):
        preprocess_fn = self.preprocess_fn

        if preprocess_fn.get("scaler"):
            train_subset, predict_subset = apply_scaler(
                train_subset, predict_subset, preprocess_fn["scaler"]
            )

        if (
            preprocess_fn.get("fourier_transform")
            and preprocess_fn["fourier_transform"]["apply"]
        ):
            ft_features = preprocess_fn["fourier_transform"]["features"]
            train_subset = apply_fourier_transform(train_subset, ft_features)
            predict_subset = apply_fourier_transform(predict_subset, ft_features)

        return train_subset, predict_subset

    def train_val_dataloader(self):
        if self.kfold is None:
            raise ValueError("Kfold options are not provided.")
        n_splits = self.kfold.get("n_splits", 5)
        shuffle = self.kfold.get("shuffle", True)
        random_state = self.kfold.get("random_state", None)

        kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        train_loaders, val_loaders = [], []

        for train_dataset in self.train_datasets:
            train_indices, val_indices = [], []

            for train_idx, val_idx in kfold.split(train_dataset):
                train_indices.append(train_idx)
                val_indices.append(val_idx)

            train_loaders.append(
                [
                    DataLoader(
                        train_dataset,
                        batch_size=self.batch_size,
                        sampler=SubsetRandomSampler(train_idx),
                        num_workers=self.num_workers,
                    )
                    for train_idx in train_indices
                ]
            )
            val_loaders.append(
                [
                    DataLoader(
                        train_dataset,
                        batch_size=self.batch_size,
                        sampler=SubsetRandomSampler(val_idx),
                        num_workers=self.num_workers,
                    )
                    for val_idx in val_indices
                ]
            )

        return train_loaders, val_loaders

    def predict_dataloader(self):
        predict_loaders = [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            for dataset in self.predict_datasets
        ]

        return predict_loaders
