import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

class DataHandler:
    def __init__(self, path):   
        self.data = pd.read_csv(path)
        self.data['genera'] = self.data.species.str.split('_').str[0]
        self.X = self.data.drop(['species', 'id', 'genera'], axis=1) 
    
    def get_y(self, label_col):
        return LabelEncoder().fit(label_col).transform(label_col) 
    
    def split_data(self, X, y, test_size, norm):
        n_splits = int(1 / test_size)
        self.n_splits = n_splits
        
        stratified_split = StratifiedKFold(n_splits=n_splits,
                                           shuffle=True,
                                           random_state=0)
  
        self.X_train_list = []
        self.X_test_list = []
        self.y_train_list = []
        self.y_test_list = []
        
        for index_train, index_test in stratified_split.split(X, y):
            X_train, X_test = X.values[index_train], X.values[index_test]
            y_train, y_test = y[index_train], y[index_test]
        
            train_size = len(X_train)
            test_size = len(X_test)
            
            if norm == 'min-max':
                min_norm_train = self.__make_array(X_train.min(axis=0), train_size)
                max_norm_train = self.__make_array(X_train.max(axis=0), train_size)            
                min_norm_test = self.__make_array(X_train.min(axis=0), test_size)
                max_norm_test = self.__make_array(X_train.max(axis=0), test_size)          
                
                X_train = self.__normalize_min_max(X_train, min_norm_train, max_norm_train)
                X_test = self.__normalize_min_max(X_test, min_norm_test, max_norm_test)
                
            elif norm == 'mean':
                mean_norm_train = self.__make_array(X_train.mean(axis=0), train_size)
                std_norm_train = self.__make_array(X_train.std(axis=0), train_size)          
                mean_norm_test = self.__make_array(X_train.mean(axis=0), test_size)
                std_norm_test = self.__make_array(X_train.std(axis=0), test_size)  
                
                X_train = self.__normalize_mean(X_train, mean_norm_train, std_norm_train)
                X_test = self.__normalize_mean(X_test, mean_norm_test, std_norm_test)            
            
            self.X_train_list.append(X_train)
            self.X_test_list.append(X_test)
            self.y_train_list.append(y_train)
            self.y_test_list.append(y_test)
    
    def __make_array(self, X, size):
        return np.array([X] * size)
    
    def __normalize_min_max(self, X, X_min, X_max):
        return (X - X_min) / (X_max - X_min)

    def __normalize_mean(self, X, X_mean, X_std):
        return (X - X_mean) / X_std
    

        