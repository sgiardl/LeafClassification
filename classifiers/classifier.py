from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class Classifier:
    """
    CLASS NAME:
        Classifier
    
    DESCRIPTION:
        Parent class for all classifying methods to be tested.
    """
    def __init__(self):
        """
        PARAMETERS:
            None.
            
        RETURNS:
            None.
            
        DESCRIPTION:
            Initalizes the object of Classifier class, with the class name
            as its self.name attribute and self.best_model and self.best_params
            as empty attributes.
        """
        
        self.name = type(self).__name__
        
        self.best_model = None
        self.best_params = None
        
    def search_hyperparameters(self, X_train, y_train, valid_size):
        """
        PARAMETERS:
            X_train : features, 2D numpy array
            y_train : labels, 1D numpy array
            valid_size : size of validation set as floating point value (ex. 0.2 for 20%)
                
        RETURNS:
            None.
            
        DESCRIPTION:
            This method uses a grid search cross-validation method
            to find the combination of hyperparameters with the 
            highest precision and saves the best model in the 
            self.best_model attribute and the best parameters
            in the self.best_params attribute.        
        """
        
        grid = GridSearchCV(self.classifier, 
                            self.param_grid, 
                            scoring='accuracy', 
                            n_jobs=-1,
                            cv=int(1 / valid_size))
        
        grid.fit(X_train, y_train)

        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_
        print(f'Best parameters: {self.best_params}')
        
    def train(self, X_train, y_train):
        """
        PARAMETERS:
            X_train : features, 2D numpy array
            y_train : labels, 1D numpy array 
            
        RETURNS:
            None.
            
        DESCRIPTION:
            This method trains the model using the self.best_model
            found in the hyperparameter search and the values
            specified in the parameters.
        """
        
        self.best_model.fit(X_train, y_train)

    def get_accuracy(self, X, y):
        """
        PARAMETERS:
            X : features, 2D numpy array
            y : labels, 1D numpy array 
            
        RETURNS:
            accuracy : training accuracy score, float %
            
        DESCRIPTION:
            This method calculates and returns the accuracy of the 
            self.best_model classifier based on the predicted labels versus
            the ground truth labels.
        """
        
        return 100 * accuracy_score(y, self.best_model.predict(X))

    def print_training_accuracy(self, X_train, y_train):
        """
        PARAMETERS:
            X_train : training features, 2D numpy array
            y_train : training labels, 1D numpy array
            
        RETURNS:
            None.
            
        DESCRIPTION:
            This method prints the training accuracy.
        """
        
        print(f'Training accuracy: {self.get_accuracy(X_train, y_train):.2f}%')
        
    def print_testing_accuracy(self, X_test, y_test):
        """
        PARAMETERS:
            X_train : testing features, 2D numpy array
            y_train : testing labels, 1D numpy array
            
        RETURNS:
            None.
            
        DESCRIPTION:
            This method prints the testing accuracy.
        """
        
        print(f'Testing accuracy: {self.get_accuracy(X_test, y_test):.2f}%')

    def print_name(self):
        """
        PARAMETERS:
            None.
            
        RETURNS:
            None.
            
        DESCRIPTION:
            This method prints the classifier name.        
        """
        
        print('=' * 40)
        print(f'Classifier Name: {self.name}\n')