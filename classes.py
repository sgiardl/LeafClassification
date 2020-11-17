




import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder

class Data_Handler:
    def __init__(self, data_train, data_test):
        self.data_train = data_train
        self.data_test = data_test

        self.prepare_data()

    def prepare_data(self):
        le = LabelEncoder().fit(self.data_train.species) 
        
        self.labels = le.transform(self.data_train.species) 
        self.data_train = self.data_train.drop(['species', 'id'], axis=1)  
        self.data_test = self.data_test.drop(['id'], axis=1)         

    def generate_data(self):
        return self.data_train, self.data_test, self.labels
        

class Classifier:
    def train(self, x, t):
        self.clf.fit(x, t)
        
    def predict(self, x):
        return self.clf.predict(x)
        
    def score(self, t, pred):
        return accuracy_score(t, pred)
    
    def error(self, t, pred):
        return np.mean((t - pred) ** 2)   
    
    def cross_validation(self, k, x, t, param_grid):
        clf_grid_search = GridSearchCV(self.clf, param_grid, cv=k,
                                       return_train_score=True,
                                       refit=True,
                                       n_jobs=-1)
        
        clf_grid_search.fit(x, t)
        
        return clf_grid_search.best_params_, clf_grid_search.best_score_, clf_grid_search.cv_results_



class SVC_Classifier(Classifier): 
    def __init__(self, C=1, kernel="linear", degree=1, gamma=1, coef0=1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        
        self.clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0) 
           
    def hp_search(self, x, t):
        param_grid = {"C": [1e-6, 1e-5],#, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
                      "kernel": ["linear", "poly", "rbf", "sigmoid"],
                      "degree": [1, 2],#, 3, 4, 5, 6, 7, 8, 9, 10]
                      "gamma": [1e-6, 1e-5],#, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
                      "coef0": [1e-6, 1e-5]#, 1e-4, 1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3]
                     }
        
        params, score, results = self.cross_validation(k=10, x=x, t=t, param_grid=param_grid)
        
        print("Parameters: " + str(params))
        print("Score: " + str(score))
        # print("Results: " + str(results))
        
        self.clf = SVC_Classifier(params)
        
        
        
        # err_min = 1e15
        
        # SVC_C = 
        # SVC_kernel = 
        # SVC_degree = 
        # SVC_gamma = 
        # SVC_coef0 = 

        # for C in SVC_C:
        #     for kernel in SVC_kernel:
        #         for degree in SVC_degree:
        #              for gamma in SVC_gamma:
        #                  for coef0 in SVC_coef0:
        #                     clf = SVC_Classifier(C=C, 
        #                                          kernel=kernel, 
        #                                          degree=degree, 
        #                                          gamma=gamma, 
        #                                          coef0=coef0)
        #                     err = clf.cross_validation(k=10, x=x, t=t)

        #                     if err < err_min:
        #                         C_optimal = C
        #                         kernel_optimal = kernel  
        #                         degree_optimal = degree
        #                         gamma_optimal = gamma
        #                         coef0_optimal = coef0
                                                                
        #                         err_min = err
                                               
        #                     print("C = " + str(C) + 
        #                           ", kernel = " + kernel + 
        #                           ", degree = " + str(degree) +
        #                           ", gamma = " + str(gamma) + 
        #                           ", coef0 = " + str(coef0) +
        #                           ", error = " + str(err))
                            
        # self.clf = SVC_Classifier(C=C_optimal, 
        #                           kernel=kernel_optimal, 
        #                           degree=degree_optimal, 
        #                           gamma=gamma_optimal, 
        #                           coef0=coef0_optimal)
                                    
        # print("OPTIMAL : C = " + str(C_optimal) +
        #       ", kernel = " + kernel_optimal +
        #       ", degree = " + str(degree_optimal) +
        #       ", gamma = " + str(gamma_optimal) +
        #       ", coef0 = " + str(coef0_optimal) +
        #       ", error = " + str(err_min))
