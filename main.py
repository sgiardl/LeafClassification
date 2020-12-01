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

from classifiers.regression import Regression
from classifiers.SVM import SVM
from classifiers.KNN import KNN
from classifiers.MLP import MLP

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
    
    clfs.append(Regression(train_data, labels, test_data, test_ids, classes))
    clfs.append(SVM(train_data, labels, test_data, test_ids, classes))  
    clfs.append(KNN(train_data, labels, test_data, test_ids, classes))  
    clfs.append(MLP(train_data, labels, test_data, test_ids, classes))  
 
    for clf in clfs:
        clf.search_hyperparameters()
        clf.train()
        clf.display_accuracies()
