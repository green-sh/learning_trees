
from learning_trees.regression import RegressionTree, get_best_split
import numpy as np
import unittest
import matplotlib.pyplot as plt

def plot_split_3():
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 2, 3])

    split, score, left_result, right_result = get_best_split(np.expand_dims(x, 0), y, 0)
    mask = x < split

    plt.clf()
    plt.scatter(x, y, label="y")
    plt.plot(x[mask], np.full(2, left_result))
    plt.plot(x[~mask], np.full(2, right_result))
    plt.savefig("images/split_3.png")

    assert left_result == 0.5
    assert right_result == 2.5

def plot_sin_regression_tree():
    x = np.expand_dims(np.linspace(-6*np.pi, 6*np.pi, 100), 0)
    y = np.sin(x[0])
    # y = 2*x[0]**2 + 3

    tree = RegressionTree(init_plot=True).train(x, y, max_deph=10, min_elements=2)

    from sklearn.tree import DecisionTreeRegressor
    tree_sklearn = DecisionTreeRegressor(max_depth=10).fit(x.T, y)

    y_hat = tree.predict(x)
    y_hat_sklearn = tree_sklearn.predict(x.T)

    plt.clf()
    plt.plot(x[0], y, label="Sin")
    plt.plot(x[0], y_hat, label="Prediction")
    plt.plot(x[0], y_hat_sklearn, label="Prediction Sklearn")
    plt.legend()
    plt.savefig("images/regression_sin.png")
    tree.graph.render("images/regression_tree_sin", format="png")

    from sklearn.metrics import mean_squared_error
    print(f"MSE: {mean_squared_error(y, y_hat)}")
    print(f"Sklearn MSE: {mean_squared_error(y, y_hat_sklearn)}")

def plot_sigmoid_regression_tree():
    x = np.expand_dims(np.linspace(-6, 6, 1000), 0)
    y = 1 / (1+np.exp(-x[0]))

    tree = RegressionTree().train(x, y, max_deph=3, min_elements=2)
    y_hat = tree.predict(x)

    plt.clf()
    plt.plot(x[0], y, label="Sigmoid")
    plt.plot(x[0], y_hat, label="Prediction")
    plt.legend()
    plt.savefig("images/regression_sigmoid.png")

def plot_graph_regression_tree():

    x = np.expand_dims(np.linspace(-6, 6, 1000), 0)
    y = 1 / (1+np.exp(-x[0]))
    tree = RegressionTree(init_plot=True).train(x, y, max_deph=10, min_elements=1)

    tree.graph.render("images/regression_tree", format="png")

# plot_graph_regression_tree()
plot_sin_regression_tree()
# plot_sigmoid_regression_tree()