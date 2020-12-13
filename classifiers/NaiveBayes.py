from sklearn.naive_bayes import GaussianNB

from classifiers.Classifier import Classifier

class NaiveBayes(Classifier):
    def __init__(self):
        super(NaiveBayes, self).__init__()
        self.classifier = GaussianNB()
        self.param_grid = {'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-4, 1e-3]
                          }
