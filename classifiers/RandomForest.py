from sklearn.ensemble import RandomForestClassifier

from classifiers.Classifier import Classifier

class RandomForest(Classifier):
    def __init__(self):
        super(RandomForest, self).__init__()
        self.classifier = RandomForestClassifier(n_jobs=-1)
        self.param_grid = {'n_estimators': [350, 400, 450],
                            'max_depth': [20, 25, 30, 35]
                           }