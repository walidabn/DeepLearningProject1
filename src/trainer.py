import numpy as np # Only to stores the results
import torch
from torch import nn

import dlc_practical_prologue as prologue


def computeErrors(model, input, target, mini_batch_size=100):
    """
    compute the numbers of errors made by the model \n
    :param model:
    :param input: Tensor of N x (2x14x14) images
    :param target: Tensor of N x 1 : (0 or 1)
    :param mini_batch_size: number of values per batch
    :return: the number of errors
    """
    err = 0
    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        if len(output) == 2:
            output = output[0]
        target_batch = target.narrow(0, b, mini_batch_size)

        for k in range(output.shape[0]):
            result = 1
            if output[k][0] < 0.5:
                result = 0
            if target_batch[k] != result:
                err += 1

    return err


def train_model(model, train_input, train_target, test_input, test_target, mini_batch_size=100, epochs=25, eta=1e-2,
                criterion=nn.BCELoss(), verbose=False):
    """
        Train a model \n
    :param model: the model
    :param train_input: Tensor of N x (2x14x14) images
    :param train_target: Tensor of N x 1: (0 or 1 for binary class)
    :param test_input: Tensor of N x (2x14x14) images
    :param test_target:  Tensor of N x 1: (0 or 1 for binary class)
    :param mini_batch_size: size of a batch :default: 100
    :param epochs:  number of epochs :default: 25
    :param eta: learning rate :default: 1e-2
    :param criterion: loss function :default: Binary Cross Entropy
    :param verbose: if true show errors for each epochs :default: False
    :return: loss, train_error, test_error
    """
    e_loss = np.zeros(epochs)
    e_train_err = np.zeros(epochs)
    e_test_err = np.zeros(epochs)

    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    for e in range(epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))

            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss = sum_loss + loss.item()
        e_loss[e] = sum_loss

        e_train_err[e] = computeErrors(model, train_input, train_target)

        e_test_err[e] = computeErrors(model, test_input, test_target)
        if verbose:
            print("epoch = ", e, "\tloss = ", sum_loss)
            print("epoch = ", e, "\ttrain_err = ", e_train_err[e])
            print("epoch = ", e, "\ttest_err = ", e_test_err[e])
            print()
        else:
            print("e = ", e, "/", epochs, end="\r")

    print("                                          ")

    return e_loss, e_train_err, e_test_err


def train_model_auxiliary_loss(model, train_input, train_target, aux_target, test_input, test_target, aux_weight=0.2,
                               mini_batch_size=100, epochs=25, eta=1e-2, criterionaux=nn.CrossEntropyLoss(),
                               criterion=nn.BCELoss(), verbose=False):
    """
        Train a model with an auxiliary loss \n
    :param model: the model
    :param train_input: Tensor of N x (2x14x14) images
    :param train_target: Tensor of N x 1: (0 or 1 for binary class)
    :param aux_target: Tensor of N x (2): the labels for both images
    :param test_input: Tensor of N x (2x14x14) images
    :param test_target:  Tensor of N x 1: (0 or 1 for binary class)
    :param aux_weight: weight of the auxiliary loss, between 0 and 1. :default: 0.2
    :param mini_batch_size: size of a batch :default: 100
    :param epochs:  number of epochs :default: 25
    :param eta: learning rate :default: 1e-2
    :param criterionaux: auxiliary loss function :default: Cross Entropy
    :param criterion: loss function :default: Binary Cross Entropy
    :param verbose: if true show errors for each epochs :default: False
    :return: loss, train_error, test_error
    """
    if aux_weight > 1.0:
        aux_weight = 1.0
    elif aux_weight < 0.0:
        aux_weight = 0.0;

    e_loss = np.zeros(epochs)
    e_train_err = np.zeros(epochs)
    e_test_err = np.zeros(epochs)

    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    for e in range(epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output, labels = model(train_input.narrow(0, b, mini_batch_size))

            loss1 = criterion(output, train_target.narrow(0, b, mini_batch_size))

            loss21 = criterionaux(labels[:, 0, :], aux_target.narrow(0, b, mini_batch_size)[:, 0])
            loss22 = criterionaux(labels[:, 1, :], aux_target.narrow(0, b, mini_batch_size)[:, 1])

            loss2 = loss21 + loss22

            loss = (1.0 - aux_weight) * loss1 + aux_weight * loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss = sum_loss + loss.item()
        e_loss[e] = sum_loss

        e_train_err[e] = computeErrors(model, train_input, train_target)

        e_test_err[e] = computeErrors(model, test_input, test_target)
        if (verbose):
            print("epoch = ", e, "\tloss = ", sum_loss)
            print("epoch = ", e, "\ttrain_err = ", e_train_err[e])
            print("epoch = ", e, "\ttest_err = ", e_test_err[e])
            print()
        else:
            print("e = ", e, "/", epochs, end="\r")

    print("                                          ")

    return e_loss, e_train_err, e_test_err


def load_and_process_data(n):
    """
        load the data and store it in a dictionnary

    :param n: number of data to load
    :return: dictionnary with the datas
    """
    train_input, train_target_dif, train_target, test_input, test_target_dif, test_target = prologue.generate_pair_sets(
        n)
    dic = {"train_input": train_input, "train_target_dif": train_target_dif.float(), "train_target_label": train_target,
           "test_input": test_input, "test_target_dif": test_target_dif.float(), "test_target_labels": test_target}

    return dic


def test_model_size(modelImage, modelDif):
    """
        test the models to see if the size of intput/output match what is expected

    :param modelImage: model
            Input: N x (14x14)
            Output: N x (10)
    :param modelDif:
        Input: N x (2 x 10)
        Output: N x (1)
    """
    x = torch.ones((9, 1, 14, 14))
    y = modelImage(x)
    if y.shape != torch.Size([9, 10]):
        print("Shape for the imageToLabel model is wrong, gets ", y.shape, " instead of ", [9, 10])
        raise
    x = torch.ones((9, 2, 10))
    y = modelDif(x)
    if y.shape != torch.Size([9, 1]):
        print("Shape for the labelsToDif model is wrong, gets ", y.shape, " instead of ", [9, 1])
        raise

