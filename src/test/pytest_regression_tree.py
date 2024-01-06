from learning_trees.decision_tree_regression import RegressionTree, get_best_split, score_split
import numpy as np
import unittest
import matplotlib.pyplot as plt

def test_score_split_3():
    x = np.array([0, 0, 0])
    y = np.array([0, 1, 2])
    score, left_result, right_result = score_split(x, y, 1)
    assert score == 1.0
    assert left_result == 0.5
    assert right_result == 1.5

def test_score_split_4():
    x = np.array([0, 0, 0, 0])
    y = np.array([0, 1, 2, 3])

    # Select 1st
    score1, left_result, right_result = score_split(x, y, 1)
    assert left_result == 0.5
    assert right_result == 2.0

    # Select 2nd
    score2, left_result, right_result = score_split(x, y, 2)
    assert left_result == 1.0
    assert right_result == 2.5

    # Because the splits are symetric score should be equal
    assert score1 == score2

test_score_split_3()
