import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class Classifier:
    def __init__(self, train_data, labels, test_data, test_data_ids, classes):
        self.train_data = train_data
        self.test_data = test_data
        self.labels = labels
        self.test_data_ids = test_data_ids
        self.classes = np.array(classes)
        self.name = type(self).__name__

        self.X_train, self.y_train, self.X_valid, self.y_valid = self.split_data()

        self.best_model = None
        self.best_score = None
        self.best_params = None

    def split_data(self):
        stratified_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
            
        for index_train, index_valid in stratified_split.split(self.train_data, self.labels):
            X_train, X_valid = self.train_data.values[index_train], self.train_data.values[index_valid]
            y_train, y_valid = self.labels[index_train], self.labels[index_valid]
            
        return X_train, y_train, X_valid, y_valid

    def search_hyperparameters(self):
        grid = GridSearchCV(self.classifier, 
                            self.param_grid, 
                            scoring='accuracy', 
                            n_jobs=-1)
        
        grid.fit(self.X_train, self.y_train)

        self.best_model = grid.best_estimator_
        self.best_score = grid.best_score_
        self.best_params = grid.best_params_
        print(f'Best parameters: {self.best_params}')
        
    def train(self):
        self.best_model.fit(self.X_train, self.y_train)

    def get_training_accuracy(self):
        return accuracy_score(self.y_train, self.best_model.predict(self.X_train))

    def get_validation_accuracy(self):
        return accuracy_score(self.y_valid, self.best_model.predict(self.X_valid))

    def print_training_accuracy(self):
        print(f'Training accuracy: {self.get_training_accuracy():.2%}')
        
    def print_validation_accuracy(self):
        print(f'Validation accuracy: {self.get_validation_accuracy():.2%}')

    def print_name(self):
        print('=' * 40)
        print(f'Classifier Name: {self.name}')
        
