import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        self.data = self.dataset[idx]
        return torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)


class PredictDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        self.data = self.dataset[idx]
        return torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)
