from sklearn.ensemble import RandomForestClassifier

from classifiers.Classifier import Classifier

class RandomForest(Classifier):
    """
    CLASS NAME:
        RandomForest
        
    DESCRIPTION:
        Child class for the RandomForest classifier, 
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
        
        super(RandomForest, self).__init__()
        self.classifier = RandomForestClassifier(n_jobs=-1)
        self.param_grid = {'n_estimators': [350, 400, 450],
                            'max_depth': [20, 25, 30, 35]
                           }