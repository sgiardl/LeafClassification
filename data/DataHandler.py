import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

class DataHandler:
    def __init__(self, path, merge_genera):   
        train_data = pd.read_csv(path)
        
        if merge_genera:
            train_data['genera'] = train_data['species'].str.split('_').str[0]

            data = LabelEncoder().fit(train_data.genera)
            self.labels = data.transform(train_data.genera) 
            
            self.train_data = train_data.drop(['species', 'id', 'genera'], axis=1)                      
        else:
            data = LabelEncoder().fit(train_data.species)
            self.labels = data.transform(train_data.species) 
            
            self.train_data = train_data.drop(['species', 'id'], axis=1)
    
    def get_full_data(self):
        return self.train_data, self.labels  
    
    def get_split_data(self, X, y, test_size):
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
            
        for index_train, index_test in stratified_split.split(X, y):
            X_train, X_test = X.values[index_train], X.values[index_test]
            y_train, y_test = y[index_train], y[index_test]
            
        return X_train, y_train, X_test, y_test