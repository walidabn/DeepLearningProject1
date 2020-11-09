import torch
from torch import nn


class DoubleImageNet(nn.Module):
    """
        takes a model for one images and apply it to 2 images
        (this class is not meant to be used externally)
    """

    def __init__(self, model):
        super(DoubleImageNet, self).__init__()
        self.net = model

    def forward(self, x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x1 = self.net(x1)
        x2 = self.net(x2)

        x = torch.cat((x1, x2), 1).view(-1, 2, 10)
        return x


class CombineNet(nn.Module):
    """
        takes:
            a model from one images to label
            a model from 2 labels to difference class
        create:
            a model that takes 2 images and output the difference
    """

    def __init__(self, modelIm, modelOut):
        super(CombineNet, self).__init__()
        self.net1 = DoubleImageNet(modelIm)
        self.net2 = modelOut

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x


class CombineWithLabels(nn.Module):
    """
        takes:
            a model from one images to label
            a model from 2 labels to difference class
        create:
            a model that takes 2 images and output the difference
            and the auxiliary labels of these images
    """

    def __init__(self, modelIm, modelOut):
        super(CombineWithLabels, self).__init__()
        self.net1 = DoubleImageNet(modelIm)
        self.net2 = modelOut

    def forward(self, x):
        labels = self.net1(x)
        out = self.net2(labels)

        return out, labels


class NotSharedCombine(nn.Module):
    """
        takes:
            a model from one images to label
            a model from 2 labels to difference class
        create:
            a model that takes 2 images and output the difference with no shared weight
    """

    def __init__(self, modelIm1, modelIm2, modelOut):
        super(NotSharedCombine, self).__init__()
        self.net1 = modelIm1
        self.net2 = modelIm2
        self.net3 = modelOut

    def forward(self, x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x1 = self.net1(x1)
        x2 = self.net2(x2)
        x = torch.cat((x1, x2), 1).view(-1, 2, 10)

        x = self.net3(x)
        return x


class NotSharedCombineWithLabels(nn.Module):
    """
        takes:
            a model from one images to label
            a model from 2 labels to difference class
        create:
            a model that takes 2 images and output the difference with no shared weight
    """

    def __init__(self, modelIm1, modelIm2, modelOut):
        super(NotSharedCombineWithLabels, self).__init__()
        self.net1 = modelIm1
        self.net2 = modelIm2
        self.net3 = modelOut

    def forward(self, x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x1 = self.net1(x1)
        x2 = self.net2(x2)
        labels = torch.cat((x1, x2), 1).view(-1, 2, 10)

        x = self.net3(labels)
        return x, labels
