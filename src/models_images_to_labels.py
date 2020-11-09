import torch
from torch import nn
from torch.nn import functional as F


class FullyConnected_I(nn.Module):
    def __init__(self):
        super(FullyConnected_I, self).__init__()
        self.fc1 = nn.Linear(14 * 14, 1000)
        self.fc2 = nn.Linear(1000, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 14 * 14)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.lin1 = nn.Linear(256, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.lin1(x.view(-1, 256)))
        x = F.relu(self.lin2(x))
        x = F.softmax(self.lin3(x),dim=1)
        return x
