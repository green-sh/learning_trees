
def test_diabetes():
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    import numpy as np

    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state=0)

    from learning_trees.regression import RegressionTree
    tree = RegressionTree().train(X_train.T, y_train, max_deph=10)

    from sklearn.tree import DecisionTreeRegressor
    tree_sklearn = DecisionTreeRegressor().fit(X_train, y_train)

    print(f"Train error: {np.mean((tree.predict(X_train.T) - y_train)**2)}")
    print(f"Test error: {np.mean((tree.predict(X_test.T) - y_test)**2)}")

    print(f"Sklearn Train error: {np.mean((tree_sklearn.predict(X_train) - y_train)**2)}")
    print(f"Sklearn Test error: {np.mean((tree_sklearn.predict(X_test) - y_test)**2)}")


    pass

test_diabetes()