# imports
import numpy as np
import torch
import torch.nn as nn


# first deep model. wide deep model based on Wide & Deep Learning for Recommender Systems, Heng-Tze Cheng, jun 2016.
# wide and deep model
class WideDeepModel(torch.nn.Module):
    def __init__(self, categorical_parts_size, continues_part_size, embedding_dim=15):
        super(WideDeepModel, self).__init__()
        np.random.seed(0)
        torch.manual_seed(0)

        self.continues_part_size = continues_part_size
        self.categorical_parts_size = categorical_parts_size

        self.r = [0]
        for i, part_size in enumerate(categorical_parts_size):
            self.r.append(self.r[i] + part_size)

        categorical_part_size = sum(categorical_parts_size)
        self.embed = nn.Embedding(num_embeddings=categorical_part_size,
                                  embedding_dim=embedding_dim)  # type: nn.Embedding
        total_dim = len(categorical_parts_size) * embedding_dim + continues_part_size
        self.deep = nn.Sequential(nn.Linear(total_dim, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Linear(1024, 512),
                                  nn.ReLU(), nn.BatchNorm1d(512), nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256),
                                  nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        first_part = x[:, -self.continues_part_size:]
        labels = []
        for i in range(len(self.r) - 1):
            labels.append(torch.argmax(x[:, self.r[i]:self.r[i + 1]], dim=1, keepdim=True) + self.r[i])
        labels = torch.cat(labels, dim=1)
        second_part = self.embed(labels)
        second_part = torch.flatten(second_part, start_dim=1)
        x = torch.cat([first_part, second_part], dim=1)
        return self.deep(x)


# second deep model. similar to the first one but add residual connection based on Deep Residual Learning for Image Recognition, Kaiming He, Dec 2015.
class ResWideDeepModel(torch.nn.Module):
    def __init__(self, categorical_parts_size, continues_part_size, embedding_dim=15):
        super(ResWideDeepModel, self).__init__()
        np.random.seed(0)
        torch.manual_seed(0)
        self.continues_part_size = continues_part_size
        self.categorical_parts_size = categorical_parts_size

        self.r = [0]
        for i, part_size in enumerate(categorical_parts_size):
            self.r.append(self.r[i] + part_size)

        categorical_part_size = sum(categorical_parts_size)
        self.embed = nn.Embedding(num_embeddings=categorical_part_size,
                                  embedding_dim=embedding_dim)  # type: nn.Embedding
        total_dim = len(categorical_parts_size) * embedding_dim + continues_part_size
        self.fc1 = nn.Sequential(nn.Linear(total_dim, 1024), nn.ReLU(), nn.BatchNorm1d(1024))
        self.fc2 = nn.Sequential(nn.Linear(total_dim + 1024, 512), nn.ReLU(), nn.BatchNorm1d(512))
        self.fc3 = nn.Sequential(nn.Linear(512 + 1024, 256), nn.ReLU(), nn.BatchNorm1d(256))
        self.fc4 = nn.Sequential(nn.Linear(512 + 256, 1), nn.Sigmoid())

    def forward(self, x):
        first_part = x[:, -self.continues_part_size:]
        labels = []
        for i in range(len(self.r) - 1):
            labels.append(torch.argmax(x[:, self.r[i]:self.r[i + 1]], dim=1, keepdim=True) + self.r[i])
        labels = torch.cat(labels, dim=1)
        second_part = self.embed(labels)
        second_part = torch.flatten(second_part, start_dim=1)
        x = torch.cat([first_part, second_part], dim=1)
        y = self.fc1(x)

        z = self.fc2(torch.cat([x, y], dim=1))
        x = y
        y = z

        z = self.fc3(torch.cat([x, y], dim=1))
        x = y
        y = z

        return self.fc4(torch.cat([x, y], dim=1))


# first nun deep model. it is a simple linear model.
# simple linear model
class LinearModel(torch.nn.Module):
    def __init__(self, categorical_parts_size, continues_part_size, embedding_dim=15):
        super(LinearModel, self).__init__()
        np.random.seed(0)
        torch.manual_seed(0)

        categorical_part_size = sum(categorical_parts_size)
        total_dim = categorical_part_size + continues_part_size
        self.net = nn.Sequential(nn.Linear(total_dim, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


# second non deep model. it is very basic and it's output doesn't depend on it's input. (constant output)
class BaseLine(nn.Module):
    def __init__(self, categorical_parts_size, continues_part_size, embedding_dim=15):
        super(BaseLine, self).__init__()
        np.random.seed(0)
        torch.manual_seed(0)

        self.ret = nn.Parameter(torch.Tensor([[-0.8]]))

    def forward(self, x):
        return torch.nn.Sigmoid()(self.ret).repeat(x.shape[0], 1)
