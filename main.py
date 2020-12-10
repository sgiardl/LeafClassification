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
from utils.AccuracyChart import AccuracyChart
from utils.FeatureChart import FeatureChart
from utils.TSNE import t_SNE

if __name__ == '__main__':
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
    title_list = ['Original Data', 
                  'X : Normalized Data (mean)',
                  'X : Normalized Data (min-max)',
                  'y : Grouping Classes by Genera', 
                  'y : Grouping Classes with TSNE']
    
    
    for i in range(len(y_list)):
        X_train, y_train, X_test, y_test = data_handler.get_split_data(X, 
                                                                       y_list[i], 
                                                                       test_size, 
                                                                       norm_list[i])
    
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
    
        accuracy_chart = AccuracyChart(names, training_acc, testing_acc, title_list[i])
        accuracy_chart.display_chart()
    
    
    
    
    
    
    
    
    
    
    
    