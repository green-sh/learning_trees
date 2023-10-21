
from learning_trees.decision_tree_regression import RegressionTree, get_best_split, score_split
import numpy as np
import unittest
import matplotlib.pyplot as plt


def plot_split_3():
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])

    score, left_result, right_result = score_split(x, y, 1)

    plt.clf()
    plt.plot(y, label="y")
    plt.plot([0, 1], np.full(2, left_result))
    plt.plot([1, 2], np.full(2, right_result))
    plt.savefig("../images/split_3.png")


def plot_split_4():
    x = np.array([0, 0, 0, 0])
    y = np.array([0, 1, 2, 3])

    # Select 1st
    score1, left_result, right_result = score_split(x, y, 1)
    
    plt.clf()
    plt.plot(y, label="y")
    plt.plot([0, 1], np.full(2, left_result), label="left split at 1")
    plt.plot([1, 2, 3], np.full(3, right_result), label="right split at 1")

    # Select 1st
    score2, left_result, right_result = score_split(x, y, 2)

    plt.plot([0, 1, 2], np.full(3, left_result), label="left split at 2")
    plt.plot([2, 3], np.full(2, right_result), label="right split at 2")
    plt.legend()
    plt.savefig("../images/split_4.png")

    print(score1 == score2)

def plot_sin_regression_tree():
    x = np.expand_dims(np.linspace(-6, 6, 100), 0)
    y = np.sin(x[0])

    tree = RegressionTree().train(x, y)
    y_hat = tree.predict(x)

    plt.clf()
    plt.plot(x[0], y, label="Sin")
    plt.plot(x[0], y_hat, label="Prediction")
    plt.legend()
    plt.savefig("../images/regression_sin.png")


def plot_sigmoid_regression_tree():
    x = np.expand_dims(np.linspace(-6, 6, 100), 0)
    y = 1 / (1+np.exp(-x[0]))

    tree = RegressionTree().train(x, y)
    y_hat = tree.predict(x)

    plt.clf()
    plt.plot(x[0], y, label="Sigmoid")
    plt.plot(x[0], y_hat, label="Prediction")
    plt.legend()
    plt.savefig("../images/regression_sigmoid.png")

plot_split_3()
plot_split_4()
plot_sin_regression_tree()
plot_sigmoid_regression_tree()