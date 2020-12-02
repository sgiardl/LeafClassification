"""
IFT712 : Techniques d'apprentissage

Automne 2020

Projet de session

Simon Giard-Leroux (12095680)
Pierre-Alexandre Dufrêne (17062312)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from classifiers.Regression import Regression
from classifiers.SupportVectorMachine import SupportVectorMachine
from classifiers.KNearestNeighbors import KNearestNeighbors
from classifiers.MultiLayerPerceptron import MultiLayerPerceptron
from classifiers.RandomForest import RandomForest
from classifiers.NaiveBayes import NaiveBayes

if __name__ == '__main__':
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    data = LabelEncoder().fit(train_data.species)
    labels = data.transform(train_data.species)
    classes = np.array(data.classes_)
    test_ids = test_data.id

    train_data = train_data.drop(['species', 'id'], axis=1)
    test_data = test_data.drop(['id'], axis=1)

    clfs = []
    names = []
    training_acc = []
    validation_acc = []
    
    clfs.append(Regression(train_data, labels, test_data, test_ids, classes))
    clfs.append(SupportVectorMachine(train_data, labels, test_data, test_ids, classes))  
    clfs.append(KNearestNeighbors(train_data, labels, test_data, test_ids, classes))  
    clfs.append(MultiLayerPerceptron(train_data, labels, test_data, test_ids, classes))  
    clfs.append(RandomForest(train_data, labels, test_data, test_ids, classes))  
    clfs.append(NaiveBayes(train_data, labels, test_data, test_ids, classes))
 
    for clf in clfs:
        clf.print_name()
        clf.search_hyperparameters()
        clf.train()
        clf.print_training_accuracy()
        clf.print_validation_accuracy()
        
        names.append(clf.name)
        training_acc.append(clf.get_training_accuracy())
        validation_acc.append(clf.get_validation_accuracy())

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, training_acc, width, label='Training')
    rects2 = ax.bar(x + width/2, validation_acc, width, label='Validation')
        
    ax.set_ylabel('Accuracy %')
    ax.set_title('Training and validation accuracies')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend() 
    fig.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    