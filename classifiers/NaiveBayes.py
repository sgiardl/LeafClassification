from sklearn.naive_bayes import GaussianNB

from classifiers.Classifier import Classifier

class NaiveBayes(Classifier):
    def __init__(self, train_data, labels, test_data, test_ids, classes):
        super(NaiveBayes, self).__init__(train_data, labels, test_data, test_ids, classes)
        self.classifier = GaussianNB()
        self.param_grid = {'var_smoothing': [1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-04, 1e-3]
                          }
