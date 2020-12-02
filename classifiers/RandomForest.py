from sklearn.ensemble import RandomForestClassifier

from classifiers.Classifier import Classifier

class RandomForest(Classifier):
    def __init__(self, X_train, y_train, X_test, y_test, valid_size):
        super(RandomForest, self).__init__(X_train, y_train, X_test, y_test, valid_size)
        self.classifier = RandomForestClassifier(n_jobs=-1)
        self.param_grid = {'n_estimators': [350, 400, 450],
                            "max_depth": [20, 25, 30, 35]
                           }