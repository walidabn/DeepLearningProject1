import matplotlib.pyplot as plt
import numpy as np


def vizualizeIm(tensor):
    plt.imshow(tensor, cmap='gray')
    plt.show()


def vizualizeIms(tensor):
    if (len(tensor.shape) == 4):
        for c in tensor:
            for im in c:
                vizualizeIm(im)
    if (len(tensor.shape) == 3):
        for im in tensor:
            vizualizeIm(im)


def plot(trainError, testError):
    epochs = np.arange(trainError.shape[0])
    fig1 = plt.figure("fig1")
    plt.title("Train and test error vs index of iteration")
    plt.xlabel("Iteration index")
    plt.ylabel("error(%)")
    plt.plot(epochs, trainError / 10, "#fc5a50", label="train error")
    plt.plot(epochs, testError / 10, label="test error")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig1.text(0.95, 0.95, "epoch = " + str(len(trainError)), fontsize=14, verticalalignment='top', bbox=props)
    plt.legend(("train error", "test error"))
    plt.show()


def plotAll(trainErrors, testErrors):
    epochs = np.arange(trainErrors[0].shape[0])
    fig1 = plt.figure("fig1")
    plt.title("Train and test error vs index of iteration")
    plt.xlabel("Iteration index")
    plt.ylabel("error(%)")
    for trainError, testError in zip(trainErrors, testErrors):
        plt.plot(epochs, trainError / 10, "#fc5a50", label="train error")
        plt.plot(epochs, testError / 10,"#000080", label="test error")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig1.text(0.95, 0.95, "epoch = " + str(len(trainError)), fontsize=14, verticalalignment='top', bbox=props)
    plt.legend(("train error", "test error"))
    plt.show()
