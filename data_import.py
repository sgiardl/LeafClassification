"""
IFT712 : Techniques d'apprentissage

Automne 2020

Projet de session

Simon Giard-Leroux (12095680)
Pierre-Alexandre Dufrêne (17062312)
"""

# Module Imports

import pandas as pd
import os
import classes as cl

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder




# Data Preparation

data_train = pd.read_csv(os.curdir + "\\Kaggle\\train.csv")
data_test = pd.read_csv(os.curdir + "\\Kaggle\\test.csv")

data_generator = cl.Data_Handler(data_train, data_test)
[data_train, data_test, labels] = data_generator.generate_data()




sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

for train_index, test_index in sss.split(data_train.values, labels):
    x_train, x_valid = data_train.values[train_index], data_train.values[test_index]
    t_train, t_valid = labels[train_index], labels[test_index]

    
# SVC

clf = cl.SVC_Classifier()
clf.hp_search(x_valid, t_valid)

clf.train(x_train, t_train)
pred = clf.predict(x_valid)
acc = clf.score(t_valid, pred)

print("Accuracy = {:.2%}".format(acc))

