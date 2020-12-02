from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class Classifier:
    def __init__(self, train_data, labels):
        self.train_data = train_data
        self.labels = labels

        self.name = type(self).__name__

        self.X_train, self.y_train, self.X_test, self.y_test = self.split_data()

        self.best_model = None
        self.best_score = None
        self.best_params = None

    def split_data(self):
        stratified_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
            
        for index_train, index_test in stratified_split.split(self.train_data, self.labels):
            X_train, X_test = self.train_data.values[index_train], self.train_data.values[index_test]
            y_train, y_test = self.labels[index_train], self.labels[index_test]
            
        return X_train, y_train, X_test, y_test

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

    def get_testing_accuracy(self):
        return accuracy_score(self.y_test, self.best_model.predict(self.X_test))

    def print_training_accuracy(self):
        print(f'Training accuracy: {self.get_training_accuracy():.2%}')
        
    def print_testing_accuracy(self):
        print(f'Testing accuracy: {self.get_testing_accuracy():.2%}')

    def print_name(self):
        print('=' * 40)
        print(f'Classifier Name: {self.name}')
        
