from sklearn.linear_model import RidgeClassifier

from classifiers.classifier import Classifier

class Regression(Classifier):
    def __init__(self, train_data, labels, test_data, test_ids, classes):
        super(Regression, self).__init__(train_data, labels, test_data, test_ids, classes)
        self.classifier = RidgeClassifier()
        self.param_grid = {'alpha': [1e-1, 2e-1, 1e-2, 1e-3, 1e-4]}
