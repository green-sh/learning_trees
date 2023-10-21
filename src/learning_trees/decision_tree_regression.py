import matplotlib.pyplot as plt
import numpy as np


def score_split(x: np.ndarray[np.number], y: np.ndarray[np.number], idx_split: int) -> np.number:
    """
    Splits array and scores result

    Note:
        data should be sorted already
    """
    left_prediction = y[idx_split:].mean()
    right_prediction = y[:idx_split].mean()

    left_error: np.ndarray[np.number] = (y[idx_split:] - left_prediction) ** 2
    right_error: np.ndarray[np.number] = (y[:idx_split] - right_prediction) ** 2

    return left_error.sum() + right_error.sum(), left_prediction, right_prediction


def score_feature(x, y, idx_feature):
    sorted_indices = np.argsort(x[idx_feature])
    sorted_x = x[idx_feature, sorted_indices]
    sorted_y = y[sorted_indices]

    best = np.infty
    best_idx = 0
    best_prediction_left = 0
    best_prediction_right = 0

    for split_idx in range(1, len(sorted_x) - 1):
        score, left_prediction, right_prediction = score_split(
            sorted_x, sorted_y, idx_split=split_idx
        )
        if score < best:
            best = score
            best_idx = split_idx - 1
            best_prediction_left = left_prediction
            best_prediction_right = right_prediction

    best_split = (x[idx_feature, best_idx] + x[idx_feature, best_idx + 1]) / 2

    return best_split, best_prediction_left, best_prediction_right


class Tree:
    def __init__(self, split_point, feature_idx, left_prediction, right_prediction) -> None:
        self.split_point = split_point
        self.feature_idx = feature_idx
        self.left_prediction = left_prediction
        self.right_prediction = right_prediction

    def predict(self, x: np.ndarray[np.number]):
        mask = x[feature_idx] < self.split_point
        res = np.empty(len(x[feature_idx]))
        res[mask == False] = self.left_prediction
        res[mask == True] = self.right_prediction

        return res


x = np.row_stack(
    [
        np.linspace(-6, 6, 100),
        np.linspace(6, -6, 100),
    ]
)  # Create one feature
y = 1 / (1 + np.exp(-x[0]))
# y = np.concatenate([np.full(50, 0), np.full(50, 1)])


feature_idx = 1
split_point, left_prediction, right_prediction = score_feature(x, y, feature_idx)
t = Tree(split_point, feature_idx, left_prediction, right_prediction)
y_hat = t.predict(x)

plt.plot(y_hat, label="Guess")
plt.plot(y, label="Real")
plt.legend()
plt.show()
