import pandas as pd

data = pd.read_csv("data.txt" , delimiter=";", header=0, index_col="Day")
data.keys()

X = data.iloc[:,:-1]

# durch alle features https://sefiks.com/2018/08/27/a-step-by-step-cart-decision-tree-example/
# Matrix mit mengen aufstellen
# Gini berchnen -> welches Feature gesplittet wird