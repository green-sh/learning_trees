import numpy as np
import pandas as pd

class Decision_Tree():
    def __init__(self) -> None:
        self.head = None
    
    def train(self, data : pd.DataFrame) -> None:
        self.head = Node()
        self.head.train(data)

    def predict(self, data : pd.DataFrame) -> list:
        if self.head:
            result_list = []
            for index, row in data.iterrows():
                result = self.head.predict(row)
                result_list.append(result)
            return result_list
        else:
            print("The model hasnt been trained yet")
    
    
class Node():
    def __init__(self) -> None:
        # feature to be used as key for this node
        self.key = None
        # dictionary with categories of the feature as key and a new predict node as value
        self.children = {}
        # category of Y that is the final output of the leaf
        self.result = None
        # no representation within train dataset
        self.out_of_dataset = False

    def train(self, data : pd.DataFrame) -> None:
        if data.empty:
            self.out_of_dataset = True
            return
        
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        # base cases
        # base case no features left
        if len(data.columns) == 1:
            self.result = y.value_counts().idxmax()
            return
        # base case one category left in y
        if y.cat.categories.size == 1:
            self.result = y.cat.categories[0]
            return

        # get best feature
        best_feature = self.get_best_feature_gini(X, y)
        self.key = best_feature

        for category in data[best_feature].cat.categories:
            new_data = data[data[best_feature] == category]
            new_data = new_data.drop(columns=best_feature)
            # refresh categories of y because they are used for determining if only one category is left
            new_data.iloc[:,-1] = new_data.iloc[:,-1].cat.remove_unused_categories()
            self.children[category] = Node()
            self.children[category].train(new_data)


    def get_best_feature_gini(self, X, y) -> str:
        """returns best_feature by calculating 
            count matrices and then calculating 
            the weighted gini index on the matrices
            to determine the best feature for splitting"""

        num_result = y.cat.categories.size
        best_feature = None
        best_feature_gini = 0
        for feature in X:
            
            num_values = X[feature].cat.categories.size

            count_matrix = np.zeros([num_values, num_result])
            
            # fill in count matrix
            for value, result in zip(X[feature].cat.codes, y.cat.codes):
                count_matrix[value][result] += 1

            # calculate gini feature and set best feature
            gini_feature = self.weighted_gini_feature(count_matrix)
            if gini_feature > best_feature_gini:
                best_feature = feature
                best_feature_gini = gini_feature

        return best_feature

    def weighted_gini_feature(self, count_matrix):
        gini_feature = 0

        for row in count_matrix:
            gini_category = 0
            # sum all the occurences of the category in respect to result category squared
            for value in row:
                gini_category += value**2
            gini_category / sum(row)
            gini_feature += gini_category

        return gini_feature

    def predict(self, data):
        """ takes a row of features and returns the prediction
            of the trained Decision Tree"""
        if self.out_of_dataset:
            raise Exception("Input Data is out of Training Dataset")

        if self.result:
            return self.result

        return self.children[data[self.key]].predict(data)
    
data = pd.read_csv("./decision_tree/data.txt" , delimiter=";", header=0, index_col="Day", dtype="category")

model = Decision_Tree()
model.train(data)
result = model.predict(data)
print(data)
print(result)