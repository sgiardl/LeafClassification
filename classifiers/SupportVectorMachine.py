from sklearn.svm import SVC

from classifiers.Classifier import Classifier

class SupportVectorMachine(Classifier):
    """
    CLASS NAME:
        SupportVectorMachine
        
    DESCRIPTION:
        Child class for the SupportVectorMachine classifier, 
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
        
        super(SupportVectorMachine, self).__init__()
        self.classifier = SVC(class_weight='balanced')
        self.param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                            'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                           }