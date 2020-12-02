from sklearn.neural_network import MLPClassifier

from classifiers.Classifier import Classifier

class MultiLayerPerceptron(Classifier):
    def __init__(self, train_data, labels, test_data, test_ids, classes):
        super(MultiLayerPerceptron, self).__init__(train_data, labels, test_data, test_ids, classes)
        self.classifier = MLPClassifier()
        self.param_grid = {'hidden_layer_sizes': [(50,), (80,), (100,)],
                            'learning_rate_init': [1e-1, 1e-2, 1e-3],
                            'solver': ['adam', 'sgd'],
                            'activation': ['relu', 'logistic']
                          }
