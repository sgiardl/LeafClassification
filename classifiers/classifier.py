from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

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
    
    def search_hyperparameters_range(self, X_train, y_train, 
                                     valid_size, param, param_grid, scale,
                                     param_grid_chosen):
        """
        PARAMETERS:
            X_train : features, 2D numpy array
            y_train : labels, 1D numpy array
            valid_size : size of validation set as floating point value (ex. 0.2 for 20%)
            param : name of the parameter being tested, string
            param_grid : dictionary of {param : list of values to test} 
            scale : scale for chart X axis, either 'log' or 'linear'
            param_grid_chosen : dictionary of {param : list of values chosen} 
            
        RETURNS:
            None.
            
        DESCRIPTION:
            This method uses a grid search cross-validation method
            to find the classifier accuracy based on a specific hyperparameter
            over a range of values and generates a chart
            showing the accuracy obtained over the range
            of input values for this parameter.
        """
        
        # Display which classifier and which parameter is being tested
        print(f'Finding hyperparameter range for classifier : {self.name}')
        print(f'and parameter : {param}\n')
        
        # Calculate number of cross-validations to perform
        cv = int(1 / valid_size)
        
        # Perform a grid search cross-validation
        # for the full range of parameter values
        grid = GridSearchCV(self.classifier, 
                            param_grid, 
                            scoring='accuracy', 
                            n_jobs=-1,
                            cv=cv,
                            return_train_score=True)
        
        grid.fit(X_train, y_train)

        # Get the parameter values, test and train scores
        param_values = [d[param] for d in grid.cv_results_['params']]
        mean_test_scores = 100 * grid.cv_results_['mean_test_score']
        mean_train_scores = 100 * grid.cv_results_['mean_train_score']

        # Perform a grid search cross-validation
        # for the chosen range of parameter values
        grid_chosen = GridSearchCV(self.classifier, 
                                    param_grid_chosen, 
                                    scoring='accuracy', 
                                    n_jobs=-1,
                                    cv=cv,
                                    return_train_score=True)
                
        grid_chosen.fit(X_train, y_train)
        
        # Get the parameter values, test and train scores
        param_values_chosen = [d[param] for d in grid_chosen.cv_results_['params']]
        mean_test_scores_chosen = 100 * grid_chosen.cv_results_['mean_test_score']
               
        # Declare chart
        fig, ax = plt.subplots()
        
        # Plot lines for full train, full test and scatter for chosen test
        plt.plot(param_values, mean_train_scores, c='blue')        
        plt.plot(param_values, mean_test_scores, c='green')
        plt.scatter(param_values_chosen, mean_test_scores_chosen, 
                    marker='x', c='red')
        
        # Show grid lines
        plt.grid(linestyle='--')
        
        # Set scale as 'linear' or 'log'
        ax.set_xscale(scale)
        
        # Show legend
        plt.legend(['Training Accuracy', 'Testing Accuracy',
                    'Values chosen'])

        # Set chart y label
        ax.set_ylabel('Mean Accuracy %')
        
        # Set chart x label
        ax.set_xlabel(param)
        
        # Set chart title
        plt.title(f'{self.name} Hyperparameter Range Search')
        
        # Show chart
        plt.show()    
        
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
        
        # Perform a grid search cross-validation
        # to find the best combination of
        # hyper-parameters
        grid = GridSearchCV(self.classifier, 
                            self.param_grid, 
                            scoring='accuracy', 
                            n_jobs=-1,
                            cv=int(1 / valid_size))
        
        grid.fit(X_train, y_train)

        # Save the best model and best parameters
        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_
        
        # Display the best parameters
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
        
        # Train the best model
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
        
        # Calculate accuracy score
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
        
        # Display training accuracy score
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
        
        # Display testing accuracy score
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
        
        # Display classifier name
        print('=' * 40)
        print(f'Classifier Name: {self.name}\n')