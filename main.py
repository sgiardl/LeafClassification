"""
IFT712 : Techniques d'apprentissage

Automne 2020

Projet de session

Simon Giard-Leroux (12095680)
"""

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

import warnings
import sys
import os

if __name__ == '__main__':
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" 
        
    test_size = 0.2
    valid_size = 0.2 # of 1 - test_size
    
    data_handler = DataHandler('data/train.csv')
    
    feature_chart = FeatureChart(data_handler.data)
    feature_chart.display_chart()
    
    X = data_handler.X
    y = data_handler.get_y(data_handler.data.species)
    y_genera = data_handler.get_y(data_handler.data.genera)
    
    t_SNE = t_SNE(X, y)
    t_SNE.display_TSNE()    
    y_t_SNE = t_SNE.y
    
    y_list = [y, y, y, y_genera, y_t_SNE]
    norm_list = ['none', 'mean', 'min-max', 'none', 'none']
    title_list = ['Method 1 : Original Data', 
                  'Method 2 : X : Normalized Data (mean)',
                  'Method 3 : X : Normalized Data (min-max)',
                  'Method 4 : y : Grouping Classes by Genera', 
                  'Method 5 : y : Grouping Classes with TSNE']
    
    for i in range(len(y_list)):
        print('=' * 40)
        print('=' * 40)
        print(f'{title_list[i]}')        
        print('=' * 40)
        print('=' * 40)
        
        data_handler.split_data(X, y_list[i], test_size, norm_list[i])
        
        n_splits = data_handler.n_splits
        
        clfs = [Ridge(),
                SupportVectorMachine(),
                KNearestNeighbors(),
                MultiLayerPerceptron(),
                RandomForest(),
                NaiveBayes()]
     
        names = []
        training_acc = []
        testing_acc = []
        
        for clf in clfs:
            clf.print_name()
            
            clf_training_acc = []
            clf_testing_acc = []            
            
            for j in range(n_splits):
                print(f'Split {j + 1} :')
                
                X_train = data_handler.X_train_list[j]
                y_train = data_handler.y_train_list[j]
                X_test = data_handler.X_test_list[j]
                y_test = data_handler.y_test_list[j]
                
                clf.search_hyperparameters(X_train, y_train, valid_size)
                clf.train(X_train, y_train)
                clf.print_training_accuracy(X_train, y_train)
                clf.print_testing_accuracy(X_test, y_test)
                
                clf_training_acc.append(clf.get_training_accuracy(X_train, y_train))
                clf_testing_acc.append(clf.get_testing_accuracy(X_test, y_test))
            
            names.append(clf.name)
            
            mean_training_acc = sum(clf_training_acc) / len(clf_training_acc)
            mean_testing_acc = sum(clf_testing_acc) / len(clf_testing_acc)
            
            print(f'\nMean training accuracy over {n_splits} splits : {mean_training_acc:.2f}%')
            print(f'Mean testing accuracy over {n_splits} splits : {mean_testing_acc:.2f}%')
            
            training_acc.append(mean_training_acc)
            testing_acc.append(mean_testing_acc)
    
        accuracy_chart = AccuracyChart(names, training_acc, testing_acc, title_list[i])
        accuracy_chart.display_chart()





    
    