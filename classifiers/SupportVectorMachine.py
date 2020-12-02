from sklearn.svm import SVC

from classifiers.Classifier import Classifier

class SupportVectorMachine(Classifier):
    def __init__(self, X_train, y_train, X_test, y_test, valid_size):
        super(SupportVectorMachine, self).__init__(X_train, y_train, X_test, y_test, valid_size)
        self.classifier = SVC()
        self.param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                            'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                           }