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
        split = (sorted_x[split_idx] + sorted_x[split_idx - 1]) / 2
        mask = sorted_x < split
        if sum(mask) == 0 or sum(~mask) == 0:
            print("wth")
            continue

        left_prediction = np.mean(sorted_y[:split_idx])
        right_prediction = np.mean(sorted_y[split_idx:])

        score = np.mean((sorted_y[:split_idx] - left_prediction) ** 2) \
            + np.mean((sorted_y[split_idx:] - right_prediction) ** 2)
        
        if score < best_score:
            best_score = score
            best_split = split
            best_prediction_left = left_prediction
            best_prediction_right = right_prediction

    if best_score is np.infty:
        raise ValueError("No split found")
    
    return best_split, best_score, best_prediction_left, best_prediction_right


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
        mask = x[self.best_feature] < self.best_split
        res = np.empty(len(x[self.best_feature]))
        res[mask == True] = self.left.predict(x[:, mask == True])
        res[mask == False] = self.right.predict(x[:, mask == False])

        return res

    def train(self, x, y, max_deph=5, min_elements=2):
        best_score = np.infty
        self.best_split = None
        best_left_prediction = None
        best_right_prediction = None
        self.best_feature = None
        for feature_idx in range(len(x)):
            split_point, score, left_prediction, right_prediction = get_best_split(
                x, y, feature_idx
            )
            if score < best_score and sum(x[feature_idx] < split_point) != 0 and sum(x[feature_idx] >= split_point) != 0:
                best_score = score
                best_left_prediction = left_prediction
                best_right_prediction = right_prediction
                self.best_split = split_point
                self.best_feature = feature_idx
        
        if self.best_feature is None:
            raise ValueError("No split found")

        if self.graph:
            self.graph.node(str(id(self)), f"{self.best_feature}\n{self.best_split:.2f}")
            if self.parent_name:
                self.graph.edge(self.parent_name, str(id(self)))

        if max_deph <= 1:
            self.left = ValueNode(best_left_prediction, graph=self.graph, parent_name=str(id(self)))
            self.right = ValueNode(best_right_prediction, graph=self.graph, parent_name=str(id(self)))
            return self

        # Do the split
        mask = x[self.best_feature] < self.best_split
        x_left, y_left = x[:, mask == True], y[mask == True]
        x_right, y_right = x[:, mask == False], y[mask == False]

        # Check if minimal minimal amounts of elements are there
        if sum(mask == True) <= min_elements:
            self.left = ValueNode(best_left_prediction, graph=self.graph, parent_name=str(id(self)))
        else:
            self.left = RegressionTree(graph=self.graph, parent_name=str(id(self))).train(x_left, y_left, max_deph=max_deph - 1, min_elements=min_elements)
            
        if sum(mask == False) <= min_elements:
            self.right = ValueNode(best_right_prediction, graph=self.graph, parent_name=str(id(self)))
        else:
            self.right = RegressionTree(graph=self.graph, parent_name=str(id(self))).train(x_right, y_right, max_deph=max_deph - 1, min_elements=min_elements)

        return self

