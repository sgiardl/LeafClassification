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

# Data Generator

data_train = pd.read_csv(os.curdir + "\\Kaggle\\train.csv")
data_test = pd.read_csv(os.curdir + "\\Kaggle\\test.csv")

data_generator = cl.Data_Handler(data_train, data_test)
[data_train, data_test, labels] = data_generator.generate_data()






    
# SVC

clf = cl.SVC_Classifier()
clf.hp_search(data_train, labels)

# clf.train(x_train, t_train)
# pred = clf.predict(x_valid)
# acc = clf.score(t_valid, pred)

# print("Accuracy = {:.2%}".format(acc))

