from classifiers.Ridge import Ridge
from classifiers.SupportVectorMachine import SupportVectorMachine
from classifiers.KNearestNeighbors import KNearestNeighbors
from classifiers.MultiLayerPerceptron import MultiLayerPerceptron
from classifiers.RandomForest import RandomForest
from classifiers.NaiveBayes import NaiveBayes

from data.DataHandler import DataHandler
from utils.AccuracyChart import AccuracyChart
from utils.FeatureChart import FeatureChart
from utils.TSNE import t_SNE

import numpy as np
import warnings
import sys
import os

if __name__ == '__main__':
    # Disable scikitlearn warnings for ConvergenceWarning
    # for multi-layer perceptron
    # (cannot be disabled any other way)
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"  
        
    # Define test and validation sizes as floating point values
    test_size = 0.2
    valid_size = 0.2 # of 1 - test_size
    
    # Set the following parameter to True to perform the
    # hyperparameter range search, else set it to False to
    # skip this part
    search_hyperparam_range = True
    
    # Load data from the data/train.csv file into the data_handler object
    data_handler = DataHandler('data/train.csv')
        
    # Generate the raw X numpy array of features
    X = data_handler.X
    
    # Generate the raw y numpy array of labels grouped by species
    y = data_handler.get_y(data_handler.data.species)
    
    if search_hyperparam_range:
        # Display that the hyperparameter range search is being performed
        print('=' * 40)
        print('Hyperparameter Range Search')
        print('=' * 40)
        
        # Find hyperparameters ranges to test for Ridge
        ridge = Ridge()
        
        param = 'alpha'
        scale = 'log'
        param_grid = {param: np.logspace(-20, 20, 100)} 
        param_grid_chosen = {param: [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]}
       
        ridge.search_hyperparameters_range(X, y, valid_size, param, 
                                           param_grid, scale, param_grid_chosen)
        
        # Find hyperparameters ranges to test for SVM
        svm = SupportVectorMachine()
        
        param = 'C'
        scale = 'log'
        param_grid = {param: np.logspace(-20, 20, 100)} 
        param_grid_chosen = {param: [1e2, 1e3, 1e4, 1e5, 1e6]}
       
        svm.search_hyperparameters_range(X, y, valid_size, param, 
                                         param_grid, scale, param_grid_chosen)    
        
        param = 'gamma'
        scale = 'log'
        param_grid = {param: np.logspace(-20, 20, 100)} 
        param_grid_chosen = {param: [2e-12, 2e-9, 3e-5, 0.1, 20]}
       
        svm.search_hyperparameters_range(X, y, valid_size, param, 
                                         param_grid, scale, param_grid_chosen)       
        
        # Find hyperparameters ranges to test for KNearestNeighbors
        knn = KNearestNeighbors()
        
        param = 'n_neighbors'
        scale = 'linear'
        param_grid = {param: np.arange(1, 100, 1)} 
        param_grid_chosen = {param: [1, 2, 3, 4, 5]}
       
        knn.search_hyperparameters_range(X, y, valid_size, param, 
                                         param_grid, scale, param_grid_chosen)    
        
        param = 'leaf_size'
        scale = 'linear'
        param_grid = {param: np.arange(10, 1000, 10)} 
        param_grid_chosen = {param: [10, 20, 30, 40, 50]}
       
        knn.search_hyperparameters_range(X, y, valid_size, param, 
                                         param_grid, scale, param_grid_chosen)       
        
        # Find hyperparameters ranges to test for MultiLayerPerceptron
        mlp = MultiLayerPerceptron()
        
        param = 'hidden_layer_sizes'
        scale = 'linear'
        param_grid = {param: np.arange(10, 110, 10)} 
        param_grid_chosen = {param: [(50,), (80,), (100,)]}
       
        mlp.search_hyperparameters_range(X, y, valid_size, param, 
                                         param_grid, scale, param_grid_chosen)       
        
        param = 'learning_rate_init'
        scale = 'log'
        param_grid = {param: np.logspace(-20, 0, 20)} 
        param_grid_chosen = {param: [1e-1, 1e-2, 1e-3]}
       
        mlp.search_hyperparameters_range(X, y, valid_size, param, 
                                         param_grid, scale, param_grid_chosen)         
        
        # Find hyperparameters ranges to test for RandomForest
        rf = RandomForest()
        
        param = 'n_estimators'
        scale = 'linear'
        param_grid = {param: np.arange(0, 550, 50)} 
        param_grid_chosen = {param: [200, 350, 450]}
       
        rf.search_hyperparameters_range(X, y, valid_size, param, 
                                        param_grid, scale, param_grid_chosen)       
    
        param = 'max_depth'
        scale = 'linear'
        param_grid = {param: np.arange(0, 105, 5)} 
        param_grid_chosen = {param: [20, 25, 30, 35]}
       
        rf.search_hyperparameters_range(X, y, valid_size, param, 
                                        param_grid, scale, param_grid_chosen)          
        
        # Find hyperparameters ranges to test for NaiveBayes
        nb = NaiveBayes()
        
        param = 'var_smoothing'
        scale = 'log'
        param_grid = {param: np.logspace(-20, 20, 100)} 
        param_grid_chosen = {param: [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2]}
       
        nb.search_hyperparameters_range(X, y, valid_size, param, 
                                        param_grid, scale, param_grid_chosen)    
        
    # Display the chart showing the variance values for each feature
    feature_chart = FeatureChart(data_handler.data)
    feature_chart.display_chart()    
    
    # Generate the y numpy array of labels grouped by genera
    y_genera = data_handler.get_y(data_handler.data.genera)
    
    # Display the t-SNE chart and generate the y numpy array of labels
    # grouped by using t-SNE
    t_SNE = t_SNE(X, y)
    t_SNE.display_TSNE()    
    y_t_SNE = t_SNE.y
    
    # Generate lists of pre-processing methods to test, combining
    # for each method a y set, a normalization method
    # and a title for the bar chart
    y_list = [y, y, y, y_genera, y_t_SNE]
    norm_list = ['none', 'mean', 'min-max', 'none', 'none']
    title_list = ['Method 1 : Original Data', 
                  'Method 2 : X : Normalized Data (mean)',
                  'Method 3 : X : Normalized Data (min-max)',
                  'Method 4 : y : Grouping Classes by Genera', 
                  'Method 5 : y : Grouping Classes with TSNE']
    
    # Generate a list of each classification method to be tested
    clfs = [Ridge(),
            SupportVectorMachine(),
            KNearestNeighbors(),
            MultiLayerPerceptron(),
            RandomForest(),
            NaiveBayes()]    
    
    # Main loop to test each pre-processing technique
    for i in range(len(y_list)):
        # Print the name of the pre-processing technique being tested
        print('=' * 40)
        print('=' * 40)
        print(f'{title_list[i]}')        
        print('=' * 40)
        print('=' * 40)
        
        # Split the data in the data_handler object into K-Fold
        # training and testing data sets
        data_handler.split_data(X, y_list[i], test_size, norm_list[i])
        n_splits = data_handler.n_splits

        # Declaring empty lists to store the names, training and
        # testing accuracies for each classifying method
        # to be tested so they can be displayed on bar charts
        names = []
        training_acc = []
        testing_acc = []
        
        # Sub-loop to test each classifying method
        for clf in clfs:
            # Print the classifying method name
            clf.print_name()
            
            # Declare empty lists to store each training
            # and testing accuracies obtained during K-Fold
            # evaluation
            clf_training_acc = []
            clf_testing_acc = []            
            
            # K-Fold loop to test each classifying method
            # n_splits times
            for j in range(n_splits):
                # Print the current split number
                print(f'Split {j + 1} :')
                
                # Get the training and testing sets from the
                # data_handler object
                X_train = data_handler.X_train_list[j]
                y_train = data_handler.y_train_list[j]
                X_test = data_handler.X_test_list[j]
                y_test = data_handler.y_test_list[j]
                
                # Perform a hyperparameter search which
                # finds the best combination of hyperparameters,
                clf.search_hyperparameters(X_train, y_train, valid_size)
                
                # Train the model one last time using these 
                # optimal hyperparameters                
                clf.train(X_train, y_train)
                
                # Display the training accuracy for the current split
                clf.print_training_accuracy(X_train, y_train)
                
                # Display the testing accuracy for the current split
                clf.print_testing_accuracy(X_test, y_test)
                
                # Add the training accuracy for the current split in the list
                clf_training_acc.append(clf.get_accuracy(X_train, y_train))
                
                # Add the testing accuracy for the current split in the list
                clf_testing_acc.append(clf.get_accuracy(X_test, y_test))
            
            # Add the classifier name to the names list
            names.append(clf.name)
            
            # Calculate the mean training accuracy over all splits
            mean_training_acc = sum(clf_training_acc) / len(clf_training_acc)
            
            # Calculate the mean testing accuracy over all splits
            mean_testing_acc = sum(clf_testing_acc) / len(clf_testing_acc)
            
            # Display the mean training accuracy over all splits
            print(f'\nMean training accuracy over {n_splits} splits : {mean_training_acc:.2f}%')
            
            # Display the mean testing accuracy over all splits
            print(f'Mean testing accuracy over {n_splits} splits : {mean_testing_acc:.2f}%')
            
            # Add the mean training accuracy to the training_acc list
            training_acc.append(mean_training_acc)
            
            # Add the mean testing accuracy to the testing_acc list
            testing_acc.append(mean_testing_acc)
    
        # Display the accuracy bar chart comparing the training and
        # testing accuracies for all classifiers for the current
        # pre-processing method being tested
        accuracy_chart = AccuracyChart(names, training_acc, testing_acc, title_list[i])
        accuracy_chart.display_chart()