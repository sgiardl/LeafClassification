from sklearn.neighbors import KNeighborsClassifier

from classifiers.Classifier import Classifier

class KNearestNeighbors(Classifier):
    """
    CLASS NAME:
        KNearestNeighbors
        
    DESCRIPTION:
        Child class for the KNearestNeighbors classifier, 
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
                
        super(KNearestNeighbors, self).__init__()
        self.classifier = KNeighborsClassifier()
        self.param_grid = {'n_neighbors': [1, 2, 3, 4, 5],
                            'weights': ['uniform', 'distance'],
                            'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                            'leaf_size': [10, 20, 30, 40, 50],
                            'p': [1, 2]
                           }