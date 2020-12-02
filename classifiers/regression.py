from sklearn.linear_model import RidgeClassifier

from classifiers.Classifier import Classifier

class Regression(Classifier):
    def __init__(self, X_train, y_train, X_test, y_test, valid_size):
        super(Regression, self).__init__(X_train, y_train, X_test, y_test, valid_size)
        self.classifier = RidgeClassifier()
        self.param_grid = {'alpha': [1e-1, 2e-1, 1e-2, 1e-3, 1e-4]}
