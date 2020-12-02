"""
IFT712 : Techniques d'apprentissage

Automne 2020

Projet de session

Simon Giard-Leroux (12095680)
Pierre-Alexandre Dufrêne (17062312)
"""

import numpy as np

import matplotlib.pyplot as plt

from classifiers.Regression import Regression
from classifiers.SupportVectorMachine import SupportVectorMachine
from classifiers.KNearestNeighbors import KNearestNeighbors
from classifiers.MultiLayerPerceptron import MultiLayerPerceptron
from classifiers.RandomForest import RandomForest
from classifiers.NaiveBayes import NaiveBayes

from data.DataHandler import DataHandler
from utils.Visualization import Visualization

if __name__ == '__main__':
    data_handler = DataHandler('data/train.csv')
    X_train, y_train, X_test, y_test = data_handler.get_split_data()
    
    clfs = [Regression(X_train, y_train, X_test, y_test),
            SupportVectorMachine(X_train, y_train, X_test, y_test),
            KNearestNeighbors(X_train, y_train, X_test, y_test),
            MultiLayerPerceptron(X_train, y_train, X_test, y_test),
            RandomForest(X_train, y_train, X_test, y_test),
            NaiveBayes(X_train, y_train, X_test, y_test)]
 
    names = []
    training_acc = []
    validation_acc = []
    
    for clf in clfs:
        clf.print_name()
        clf.search_hyperparameters()
        clf.train()
        clf.print_training_accuracy()
        clf.print_testing_accuracy()
        
        names.append(clf.name)
        training_acc.append(clf.get_training_accuracy() * 100)
        validation_acc.append(clf.get_testing_accuracy() * 100)

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, training_acc, width, label='Training')
    rects2 = ax.bar(x + width/2, validation_acc, width, label='Testing')
        
    ax.set_ylabel('Accuracy %')
    ax.set_title('Training and testing accuracies')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation='vertical')
    ax.legend(bbox_to_anchor=(1.05, 1)) 
    plt.ylim([90, 100])
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    