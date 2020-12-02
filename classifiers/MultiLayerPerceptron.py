from sklearn.neural_network import MLPClassifier
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

from classifiers.Classifier import Classifier

class MultiLayerPerceptron(Classifier):
    def __init__(self, X_train, y_train, X_test, y_test):
        simplefilter("ignore", category=ConvergenceWarning)
        super(MultiLayerPerceptron, self).__init__(X_train, y_train, X_test, y_test)
        self.classifier = MLPClassifier()
        self.param_grid = {'hidden_layer_sizes': [(50,), (80,), (100,)],
                            'learning_rate_init': [1e-1, 1e-2, 1e-3],
                            'solver': ['adam', 'sgd'],
                            'activation': ['relu', 'logistic']
                          }
