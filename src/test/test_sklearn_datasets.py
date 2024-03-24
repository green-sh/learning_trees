
def test_regression_diabetes():
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    import numpy as np

    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state=0)

    from learning_trees.regression import RegressionTree
    tree = RegressionTree(init_plot=True).train(X_train.T, y_train, max_deph=20, min_elements=1)
    print(f"Train error: {np.mean((tree.predict(X_train.T) - y_train)**2)}")
    print(f"Test error: {np.mean((tree.predict(X_test.T) - y_test)**2)}")

    print(tree.graph.render("images/regression_tree", format="png"))

    from sklearn.tree import DecisionTreeRegressor
    tree_sklearn = DecisionTreeRegressor(max_depth=20).fit(X_train, y_train)
    print(f"Sklearn Train error: {np.mean((tree_sklearn.predict(X_train) - y_train)**2)}")
    print(f"Sklearn Test error: {np.mean((tree_sklearn.predict(X_test) - y_test)**2)}")
    pass

def test_classifcation_tree():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    import numpy as np

    breast_cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, random_state=0)

    pass


test_regression_diabetes()
# test_classifcation_tree()
