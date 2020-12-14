from sklearn.naive_bayes import GaussianNB

from classifiers.Classifier import Classifier

class NaiveBayes(Classifier):
    """
    CLASS NAME:
        NaiveBayes
        
    DESCRIPTION:
        Child class for the NaiveBayes classifier, 
        inherits from the Classifier parent class.
    """
    def __init__(self):
        """
        PARAMETERS:
            None.
            
        RETURNS:
            None.
            
        DESCRIPTION:
            Initializes the class with the range of parameters
            to test during hyperparameter search in the
            self.param_grid attribute. The sklearn classifier 
            class is specified in the self.classifier attribute.
        """
        
        super(NaiveBayes, self).__init__()
        self.classifier = GaussianNB()
        self.param_grid = {'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-4, 1e-3]
                          }