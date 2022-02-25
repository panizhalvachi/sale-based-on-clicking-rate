import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
)
from pandas import DataFrame


class RecomenderDataset(Dataset):
    def __init__(self, df: DataFrame):
        super(RecomenderDataset, self).__init__()

        x = df.loc[:, df.columns != "Sale"].values
        y = df.loc[:, "Sale"].values

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, item):
        return self.x_train[item], self.y_train[item]


class RecomenderTestingDataset(Dataset):
    def __init__(self, df: DataFrame):
        super(RecomenderTestingDataset, self).__init__()

        x = df.loc[:, df.columns != "Sale"].values

        self.x_train = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, item):
        return self.x_train[item]


def build_dataloader(data, batch_size=6):
    dataset = RecomenderDataset(data)
    random_sampler = RandomSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        sampler=random_sampler,
        drop_last=False,
        pin_memory=True)
    return dataloader
