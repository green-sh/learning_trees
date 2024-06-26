import matplotlib.pyplot as plt
import numpy as np

def score_split(x: np.ndarray[np.number], y: np.ndarray[np.number], idx_split: int):
    """
    Splits array and scores result

    Note:
        - data should be sorted already
        - there should be at least 2 elements in y
    """
    left_prediction = y[:idx_split+1].mean()
    right_prediction = y[idx_split:].mean()

    left_error: np.ndarray[np.number] = (y[:idx_split+1] - left_prediction) ** 2
    right_error: np.ndarray[np.number] = (y[idx_split:] - right_prediction) ** 2

    return left_error.sum() + right_error.sum(), left_prediction, right_prediction

def get_best_split(x, y, idx_feature):
    """
    Iterate through all split points and return best split

    The dataset d is devided into 2 sets s1, s2 with a given split point p where:
    - s1 is the subset of d of all elements smaller then the splitpoint p
    - s2 is the subset of d of all elements bigger then the splitpoint p

    then we predict 
    - mean(s1) if x is smaller than p or
    - mean(s2) if x is bigger then p
    
    p is determined by brute force
    1. go through every data point
        1. split data at this point
        2. evaluate split
    2. return best split point (and the score and predictions it made)

    A few optimization ideas:
    - choose random splits 
    - choose split in a given intervall
    """
    sorted_indices = np.argsort(x[idx_feature])
    sorted_x = x[idx_feature, sorted_indices]
    sorted_y = y[sorted_indices]

    best_score = np.infty
    best_idx = 0
    best_prediction_left = 0
    best_prediction_right = 0

    for split_idx in range(1, len(sorted_x) - 1):
        score, left_prediction, right_prediction = score_split(
            sorted_x, sorted_y, idx_split=split_idx
        )
        if score < best_score:
            best_score = score
            best_idx = split_idx - 1
            best_prediction_left = left_prediction
            best_prediction_right = right_prediction

    best_split = (x[idx_feature, best_idx] + x[idx_feature, best_idx + 1]) / 2

    return best_split, best_score, best_prediction_left, best_prediction_right


class ValueNode:
    def __init__(self, value) -> None:
        self.value = value

    def predict(self, x):
        return np.full(x.shape[1], self.value)

class RegressionTree:
    def __init__(self) -> None:
        pass

    def predict(self, x: np.ndarray[np.number]):
        mask = x[self.best_feature] < self.best_split
        res = np.empty(len(x[self.best_feature]))
        res[mask == True] = self.left.predict(x[:, mask == True])
        res[mask == False] = self.right.predict(x[:, mask == False])

        return res

    def train(self, x, y, max_deph=5, min_elements=2):
        # Choose best feature and split
        # TODO code dupplication with above
        # Maybe create a dataclass with scored value equality value
        best_score = np.infty
        self.best_split = 0
        best_left_prediction = 0
        best_right_prediction = 0
        self.best_feature = 0
        for feature_idx in range(len(x)):
            split_point, score, left_prediction, right_prediction = get_best_split(
                x, y, feature_idx
            )
            if score < best_score:
                best_score = score
                best_left_prediction = left_prediction
                best_right_prediction = right_prediction
                self.best_split = split_point
                self.best_feature = feature_idx

        if max_deph == 0:
            self.left = ValueNode(best_left_prediction)
            self.right = ValueNode(best_right_prediction)
            return self

        # Do the split
        mask = x[self.best_feature] < self.best_split
        x_left, y_left = x[:, mask == True], y[mask == True]
        x_right, y_right = x[:, mask == False], y[mask == False]

        # Check if minimal minimal amounts of elements are there
        if sum(mask == True) <= min_elements:
            self.left = ValueNode(best_left_prediction)
        else:
            self.left = RegressionTree().train(x_left, y_left, max_deph=max_deph - 1)

        if sum(mask == False) <= min_elements:
            self.right = ValueNode(best_right_prediction)
        else:
            self.right = RegressionTree().train(x_right, y_right, max_deph=max_deph - 1)

        return self