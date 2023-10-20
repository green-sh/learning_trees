import numpy as np

class Decision_Tree():
    def __init__(self) -> None:
        self.head = None
    
    def train(self, data : np.array):
        # takes numpy matrix as input with most right column as Y and the rest of the columns as X

        # store features as a array of indexes
        features = [i for i in range(len(data[0]) - 1)]
        Y = (len(data[0]) - 1)
        
        # temporal array for storing the different categories of a column
        category_sets = [set() for column in range(len(data[0]))]

        # fill in the sets to track the different categories
        for column_index in range(len(data[0])):
            for row_index in range(len(data)):
                category_sets[column_index].add(data[row_index][column_index])
        
        # create category dictionaries out of the sets
        categories_dicts = [i for i in range(len(data[0]))]
        for column in range(len(data[0])):
            list_of_categories = list(category_sets[column])
            categories_dicts[column] = dict([[list_of_categories[i], i] for i in range(len(list_of_categories))])

        print(categories_dicts)
        # creation of the count matrices
        # feature categories as rows and Y categories as columns
        arr_count_matrices = [None for i in range(len(data[0]) - 1)]
        for feature in features:
            arr_count_matrices[feature] = np.zeros([len(categories_dicts[feature]), len(categories_dicts[Y])])

        for feature in features:
            cat_dict = categories_dicts[feature]
            Y_cat_dict = categories_dicts[Y]
            for row_index in range(len(data)):
                arr_count_matrices[feature][cat_dict[data[row_index][feature]]][Y_cat_dict[data[row_index][Y]]] += 1

        self.head = Node()
        self.head.train(data=data, 
                        features=features,
                        cat_dicts=categories_dicts,
                        count_matrices=arr_count_matrices)

    def predict(self):
        return self.head.predict()
    
    
class Node():
    def __init__(self) -> None:
        # feature to be used as key as numeric index
        self.key = None
        # dictionary with categories of the feature as key and a new predict node as value
        self.children = {}
        # category of Y that is the final output of the leaf
        self.result = None

    def train(self, data, features, cat_dicts, count_matrices):
        # implement base case

        # get best feature by gini and set it as key for the self.predict_node
        best_feature = self.get_best_feature_gini_based(features, count_matrices)
        self.key = cat_dicts[best_feature]

        # Y Index
        Y = (len(data[0]) - 1)

        # remove best feature because we dont have to consider it anymore
        features.remove(best_feature)

        # initialise new count_matrices, one for every category of the best feature
        new_count_matrices = [None for _ in count_matrices]
        cat_count_matrices = [new_count_matrices.copy() for category in cat_dicts[best_feature]]
        for i in range(len(cat_count_matrices)):
            for feature in features:
                cat_count_matrices[i][feature] = np.zeros([len(cat_dicts[feature]), len(cat_dicts[Y])])
        
        arr_new_data = [list() for category in cat_dicts[best_feature]]

        # filter data and fill in new count matrices
        for row in data:
            best_feature_category = cat_dicts[row[best_feature]] # as Index
            arr_new_data[best_feature_category].append(row)
            for feature in features:
                # for the count matrix
                row_position = cat_dicts[feature][row[feature]]
                col_position = cat_dicts[Y][row[Y]]

                cat_count_matrices[best_feature_category][feature][row_position][col_position] += 1

        # initialise and train new nodes
        for category in cat_dicts[best_feature]:
            node = self.children[category] = Node()
            category_index = cat_dicts[best_feature][category]

            node.train(data = arr_new_data[category_index],
                       features=features.copy(),
                       cat_dicts=cat_dicts,
                       count_matrices = cat_count_matrices[category_index])

    def get_best_feature_gini_based(self, features, count_matrices):
        pass

    def predict(self, data):
        # takes a row of features and returns the prediction 
        if self.result:
            return self.result

        return self.children[data[self.key]].predict(data)
    

test_array = np.array([["a", "b", "c", 6, "yes"], [3, 4, 5, 6, "yes"], [3, 4, 5, 6, "Yes"]])

Decision_Tree().train(test_array)
