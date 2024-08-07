from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
class SVM():
    
    classifier = None
    
    def __init__(self):
        self.classifier = SVC(kernel = 'rbf', class_weight='balanced')
    
    def classify_fold_accuracy(self, X, y):
        
        #print(np.sum(y))
        
        #self.classifier.fit(X, y)
        print(f'number of classes {max(y) + 1}')
        
        return np.mean(cross_val_score(self.classifier, X, y, cv=5))

    def classify(self, X,y):
        return self.classify_fold_accuracy(X,y)

     

