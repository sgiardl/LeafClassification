from sklearn.neighbors import KNeighborsClassifier

from classifiers.Classifier import Classifier

class KNearestNeighbors(Classifier):
    def __init__(self, X_train, y_train, X_test, y_test, valid_size):
        super(KNearestNeighbors, self).__init__(X_train, y_train, X_test, y_test, valid_size)
        self.classifier = KNeighborsClassifier()
        self.param_grid = {'n_neighbors': [1, 2, 3, 4, 5],
                            'weights': ['uniform', 'distance'],
                            'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                            'leaf_size': [10, 20, 30, 40, 50],
                            'p': [1, 2]
                           }
