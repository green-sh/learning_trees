
from learning_trees.decision_tree_regression import RegressionTree, get_best_split, score_split
import numpy as np
import unittest
import matplotlib.pyplot as plt


def plot_split_3():
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])

    score, left_result, right_result = score_split(x, y, 1)

    plt.plot(y, label="y")
    plt.plot([0, 1], np.full(2, left_result))
    plt.plot([1, 2], np.full(2, right_result))
    plt.show()


def plot_split_4():
    x = np.array([0, 0, 0, 0])
    y = np.array([0, 1, 2, 3])

    # Select 1st
    score1, left_result, right_result = score_split(x, y, 1)

    plt.plot(y, label="y")
    plt.plot([0, 1], np.full(2, left_result))
    plt.plot([1, 2, 3], np.full(3, right_result))
    plt.show()


    # Select 1st
    score2, left_result, right_result = score_split(x, y, 2)

    plt.plot(y, label="y")
    plt.plot([0, 1, 2], np.full(3, left_result))
    plt.plot([2, 3], np.full(2, right_result))
    plt.show()

    print(score1 == score2)

plot_split_4()