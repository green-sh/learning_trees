from learning_trees.regression import RegressionTree, get_best_split, score_split
import numpy as np
import unittest
import matplotlib.pyplot as plt

class RegressionTreeTest(unittest.TestCase):

    def test_score_split_3(self):
        x = np.array([0, 0, 0])
        y = np.array([0, 1, 2])
        score, left_result, right_result = score_split(x, y, 1)
        self.assertAlmostEqual(score, 1.0)
        self.assertAlmostEqual(left_result, 0.5)
        self.assertAlmostEqual(right_result, 1.5)

    def test_score_split_4(self):
        x = np.array([0, 0, 0, 0])
        y = np.array([0, 1, 2, 3])

        # Select 1st
        score1, left_result, right_result = score_split(x, y, 1)
        self.assertAlmostEqual(left_result, 0.5)
        self.assertAlmostEqual(right_result, 2.0)

        # Select 2nd
        score2, left_result, right_result = score_split(x, y, 2)
        self.assertAlmostEqual(left_result, 1.0)
        self.assertAlmostEqual(right_result, 2.5)

        # Because the splits are symetric score should be equal
        self.assertAlmostEqual(score1, score2)



if __name__=="__main__":
    unittest.main()
