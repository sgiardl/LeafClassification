from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class Classifier:
    def __init__(self):
        self.name = type(self).__name__
        
        self.best_model = None
        self.best_score = None
        self.best_params = None
        
    def search_hyperparameters(self, X_train, y_train, valid_size):
        grid = GridSearchCV(self.classifier, 
                            self.param_grid, 
                            scoring='accuracy', 
                            n_jobs=-1,
                            cv=int(1 / valid_size))
        
        grid.fit(X_train, y_train)

        self.best_model = grid.best_estimator_
        self.best_score = grid.best_score_
        self.best_params = grid.best_params_
        print(f'Best parameters: {self.best_params}')
        
    def train(self, X_train, y_train):
        self.best_model.fit(X_train, y_train)

    def get_training_accuracy(self, X_train, y_train):
        return 100 * accuracy_score(y_train, self.best_model.predict(X_train))

    def get_testing_accuracy(self, X_test, y_test):
        return 100 * accuracy_score(y_test, self.best_model.predict(X_test))

    def print_training_accuracy(self, X_test, y_test):
        print(f'Training accuracy: {self.get_training_accuracy(X_test, y_test):.2f}%')
        
    def print_testing_accuracy(self, X_test, y_test):
        print(f'Testing accuracy: {self.get_testing_accuracy(X_test, y_test):.2f}%')

    def print_name(self):
        print('=' * 40)
        print(f'Classifier Name: {self.name}\n')
        
