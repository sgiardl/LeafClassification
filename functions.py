import numpy as np
from sklearn.metrics import accuracy_score#, log_loss

def testClassifier(clf, X_train, y_train, X_valid, y_valid):
    clfName = clf.__class__.__name__
    
    clf.fit(X_train, y_train)
    
    pred = clf.predict(X_valid)
    acc = accuracy_score(y_valid, pred)

    print(clfName + " : Accuracy = {:.2%}".format(acc))

def crossValidation(k, err, x_split, t_split):
    for j in range(0, k, 1):   
        # Entraîner sur X_split[tout sauf j], t_split[tout sauf j]                                   
        x_train = x_split[:j] + x_split[j+1:]
        t_train = t_split[:j] + t_split[j+1:]
        
        x_train = np.array([i for sl in x_train for i in sl])
        t_train = np.array([i for sl in t_train for i in sl])
        
        self.entrainement(x_train, t_train)
        
        # Tester sur X_split[j], t_split[j]
        x_test = np.array(x_split[j])
        t_test = np.array(t_split[j])
 
        t_pred = np.array([self.prediction(x) for x in x_test])
        
        # Calculer l'erreur moyenne sur les k données
        err[j] = self.erreur(t_test,t_pred)

    # Calculer l'erreur moyenne sur toutes les cross-validations
    return np.mean(err)
