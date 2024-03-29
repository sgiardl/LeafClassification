from sklearn.linear_model import RidgeClassifier

from classifiers.Classifier import Classifier

class Ridge(Classifier):
    """
    CLASS NAME:
        Ridge
        
    DESCRIPTION:
        Child class for the Ridge classifier, 
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
        
        super(Ridge, self).__init__()
        self.classifier = RidgeClassifier()
        self.param_grid = {'alpha': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]}