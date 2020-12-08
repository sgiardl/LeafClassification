"""
IFT712 : Techniques d'apprentissage

Automne 2020

Projet de session

Simon Giard-Leroux (12095680)
"""

from classifiers.LogisticRegression import LogisticRegression
from classifiers.SupportVectorMachine import SupportVectorMachine
from classifiers.KNearestNeighbors import KNearestNeighbors
from classifiers.MultiLayerPerceptron import MultiLayerPerceptron
from classifiers.RandomForest import RandomForest
from classifiers.NaiveBayes import NaiveBayes

from data.DataHandler import DataHandler
from utils.Chart import Chart
from utils.TSNE import t_SNE

if __name__ == '__main__':
    test_size = 0.2
    valid_size = 0.2
    merge_genera = True
    
    data_handler = DataHandler('data/train.csv', merge_genera)
    X, y = data_handler.get_full_data()
    
    t_SNE = t_SNE(X, y)
    y_t_SNE = t_SNE.display_TSNE()    
    
    y_list = [y, y_t_SNE]
    title_list = ['No Grouping', 'Grouping with TSNE']
    
    for i in range(len(y_list)):
        X_train, y_train, X_test, y_test = data_handler.get_split_data(X, y_list[i], test_size)
    
        clfs = [LogisticRegression(X_train, y_train, X_test, y_test, valid_size),
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
    
        chart = Chart(names, training_acc, testing_acc, title_list[i])
        chart.display_chart()
    
    
    
    
    
    
    
    
    
    
    
    