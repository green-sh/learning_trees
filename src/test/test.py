import pandas as pd
import numpy as np

data = pd.read_csv("data/data.txt" , delimiter=";", header=0, index_col="Day", dtype="category")
data.keys()

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# durch alle features https://sefiks.com/2018/08/27/a-step-by-step-cart-decision-tree-example/
# Matrix mit mengen aufstellen
# Gini berchnen -> welches Feature gesplittet wird

# Gini Matrix per feature
# Category.cat.codes gives back integer value of category value
arr = []
num_result = y.cat.categories.size 
for feature in X:
    print(feature)
    
    num_values = X[feature].cat.categories.size

    gini_matrix = np.zeros([num_values, num_result])
    
    for value, result in zip(X[feature].cat.codes, y.cat.codes):
        gini_matrix[value][result] += 1

    print(feature, gini_matrix)
    arr.append(gini_matrix)