from sklearn.linear_model import RidgeClassifier

from classifiers.Classifier import Classifier

class Ridge(Classifier):
    def __init__(self):
        super(Ridge, self).__init__()
        self.classifier = RidgeClassifier()
        self.param_grid = {'alpha': [1e-1, 2e-1, 1e-2, 1e-3, 1e-4]}
