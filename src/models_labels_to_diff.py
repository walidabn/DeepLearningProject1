import torch
from torch import nn
from torch.nn import functional as F


class FullyConnected_II(nn.Module):
    def __init__(self):
        super(FullyConnected_II, self).__init__()
        self.fc1 = nn.Linear(20, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 20)))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class ArgMax(nn.Module):
    def __init__(self):
        super(ArgMax, self).__init__()

    def forward(self, x):
        x = x.argmax(dim=2)
        x = x[:, 0] - x[:, 1]
        x = -x.sign()
        x = (x + 2) / 2
        x = x.float()
        return x.view(-1, 1)
