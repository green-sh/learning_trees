import matplotlib.pyplot as plt
import numpy as np

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
    best_prediction_left = None
    best_prediction_right = None

    for split_idx in range(1, len(sorted_x)):
        x_value = sorted_x[split_idx]
        # iterate to get to the last sample of a certain value in a feature
        if split_idx < (len(sorted_x) - 1) and x_value == sorted_x[split_idx + 1]:
            # print("wth")
            continue

        left_prediction = np.mean(sorted_y[:split_idx])
        right_prediction = np.mean(sorted_y[split_idx:])

        score = ((sorted_y[:split_idx] - left_prediction) ** 2).sum() \
            + ((sorted_y[split_idx:] - right_prediction) ** 2).sum()
        
        if score < best_score:
            best_score = score
            best_split_idx = split_idx
            best_split_value = sorted_x[best_split_idx]
            best_prediction_left = left_prediction
            best_prediction_right = right_prediction

    if best_score is np.infty:
        raise ValueError("No split found")
    
    return best_split_value, best_score, best_prediction_left, best_prediction_right


class ValueNode:
    def __init__(self, value, graph = None, parent_name = None) -> None:
        self.value = value
        if graph:
            graph.node(str(id(self)), f"Value:\n{value:.2f}")
            if parent_name:
                graph.edge(parent_name, str(id(self)))

    def predict(self, x):
        return np.full(x.shape[1], self.value)

# TODO: implement Node class instead of ValueNode and RegressionTree
class RegressionTree:
    def __init__(self, init_plot = False, graph = None, parent_name=None) -> None:
        self.graph = graph
        self.parent_name = parent_name
        if init_plot:
            import graphviz
            self.graph = graphviz.Digraph()

    def predict(self, x: np.ndarray[np.ndarray[np.number]]):
        mask = x[self.best_feature] < self.best_split_value
        res = np.empty(len(x[self.best_feature]))
        res[mask == True] = self.left.predict(x[:, mask == True])
        res[mask == False] = self.right.predict(x[:, mask == False])

        return res

    def train(self, x, y, max_depth=5, min_elements=2):
        best_score = np.infty
        self.best_split_idx = None
        self.best_split_value = None
        best_left_prediction = None
        best_right_prediction = None
        self.best_feature = None
        for feature_idx in range(len(x)):
            split_value, score, left_prediction, right_prediction = get_best_split(
                x, y, feature_idx
            )
            if score < best_score and sum(x[feature_idx] < split_value) != 0 and sum(x[feature_idx] >= split_value) != 0:
                best_score = score
                best_left_prediction = left_prediction
                best_right_prediction = right_prediction
                self.best_split_value = split_value
                self.best_feature = feature_idx
        
        if self.best_feature is None:
            raise ValueError("No split found")

        if self.graph:
            self.graph.node(str(id(self)), f"{self.best_feature}\n{self.best_split_value:.2f}")
            if self.parent_name:
                self.graph.edge(self.parent_name, str(id(self)))

        # base case max depth reached
        if max_deph <= 1:
            self.left = ValueNode(best_left_prediction, graph=self.graph, parent_name=str(id(self)))
            self.right = ValueNode(best_right_prediction, graph=self.graph, parent_name=str(id(self)))
            return self

        mask = x[self.best_feature] < self.best_split_value
        # Do the split
        x_left, y_left = x[:, mask], y[mask]
        x_right, y_right = x[:, ~mask], y[~mask]

        # base case less than {min_elements} unique values left in x
        if len(np.unique(x_left)) <= min_elements:
            self.left = ValueNode(best_left_prediction, graph=self.graph, parent_name=str(id(self)))
        else:
            self.left = RegressionTree(graph=self.graph, parent_name=str(id(self))).train(x_left, y_left, max_deph=max_deph - 1, min_elements=min_elements)
            
        if len(np.unique(x_right)) <= min_elements:
            self.right = ValueNode(best_right_prediction, graph=self.graph, parent_name=str(id(self)))
        else:
            self.right = RegressionTree(graph=self.graph, parent_name=str(id(self))).train(x_right, y_right, max_deph=max_deph - 1, min_elements=min_elements)

        return self
