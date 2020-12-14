from sklearn.neural_network import MLPClassifier

from classifiers.Classifier import Classifier

class MultiLayerPerceptron(Classifier):
    """
    CLASS NAME:
        MultiLayerPerceptron
        
    DESCRIPTION:
        Child class for the MultiLayerPerceptron classifier, 
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
                
        super(MultiLayerPerceptron, self).__init__()
        self.classifier = MLPClassifier()
        self.param_grid = {'hidden_layer_sizes': [(50,), (80,), (100,)],
                            'learning_rate_init': [1e-1, 1e-2, 1e-3],
                            'solver': ['adam', 'sgd'],
                            'activation': ['relu', 'logistic']
                          }