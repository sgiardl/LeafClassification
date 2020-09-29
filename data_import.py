"""
IFT712 : Techniques d'apprentissage

Automne 2020

Projet de session

Simon Giard-Leroux (12095680)
Pierre-Alexandre Dufrêne (17062312)
"""

# Importation des modules
import pandas as pd
import os

# Lecture des fichiers CSV pour les ensembles d'entraînement et de test
train_data = pd.read_csv(os.curdir + "\\Kaggle\\train.csv")
test_data = pd.read_csv(os.curdir + "\\Kaggle\\test.csv")

