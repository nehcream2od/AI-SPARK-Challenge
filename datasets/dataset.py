import torch
from base.base_dataset import PredictDataset, TrainDataset


class CustomTrainDataset(TrainDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    def __getitem__(self, idx):
        self.data = self.dataset[idx]
        return torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)


class CustomPredictDataset(PredictDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    def __getitem__(self, idx):
        self.data = self.dataset[idx]
        return torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)
