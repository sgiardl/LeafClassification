from sklearn.neighbors import KNeighborsClassifier

from classifiers.classifier import Classifier

class KNN(Classifier):
    def __init__(self, train_data, labels, test_data, test_ids, classes):
        super(KNN, self).__init__(train_data, labels, test_data, test_ids, classes)
        self.classifier = KNeighborsClassifier()
        self.param_grid = {'n_neighbors': [1, 2, 3, 4, 5],
                            'weights': ['uniform', 'distance'],
                            'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                            'leaf_size': [10, 20, 30, 40, 50],
                            'p': [1, 2]
                           }
