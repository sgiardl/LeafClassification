"""
IFT712 : Techniques d'apprentissage

Automne 2020

Projet de session

Simon Giard-Leroux (12095680)
Pierre-Alexandre Dufrêne (17062312)
"""

from classifiers.Regression import Regression
from classifiers.SupportVectorMachine import SupportVectorMachine
from classifiers.KNearestNeighbors import KNearestNeighbors
from classifiers.MultiLayerPerceptron import MultiLayerPerceptron
from classifiers.RandomForest import RandomForest
from classifiers.NaiveBayes import NaiveBayes

from data.DataHandler import DataHandler
from utils.Visualization import Visualization

if __name__ == '__main__':
    test_size = 0.2
    valid_size = 0.2
    merge_genera = True
    
    data_handler = DataHandler('data/train.csv', test_size, merge_genera)
    X_train, y_train, X_test, y_test = data_handler.get_split_data()
    
    clfs = [Regression(X_train, y_train, X_test, y_test, valid_size),
            SupportVectorMachine(X_train, y_train, X_test, y_test, valid_size),
            KNearestNeighbors(X_train, y_train, X_test, y_test, valid_size),
            MultiLayerPerceptron(X_train, y_train, X_test, y_test, valid_size),
            RandomForest(X_train, y_train, X_test, y_test, valid_size),
            NaiveBayes(X_train, y_train, X_test, y_test, valid_size)]
 
    names = []
    training_acc = []
    testing_acc = []
    
    for clf in clfs:
        clf.print_name()
        clf.search_hyperparameters()
        clf.train()
        clf.print_training_accuracy()
        clf.print_testing_accuracy()
        
        names.append(clf.name)
        training_acc.append(clf.get_training_accuracy() * 100)
        testing_acc.append(clf.get_testing_accuracy() * 100)

    chart = Visualization(names, training_acc, testing_acc)
    chart.display_chart()
    
    
    
    
    
    
    
    
    
    
    
    