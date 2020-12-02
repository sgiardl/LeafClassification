import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

class DataHandler:
    def __init__(self, path):   
        train_data = pd.read_csv(path)
        self.train_data = train_data.drop(['species', 'id'], axis=1)
        
        data = LabelEncoder().fit(train_data.species)
        self.labels = data.transform(train_data.species) 
        
    def get_split_data(self):
        stratified_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
            
        for index_train, index_test in stratified_split.split(self.train_data, self.labels):
            X_train, X_test = self.train_data.values[index_train], self.train_data.values[index_test]
            y_train, y_test = self.labels[index_train], self.labels[index_test]
            
        return X_train, y_train, X_test, y_test