from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
class SVM():
    
    classifier = None
    
    def __init__(self):
        self.classifier = SVC(kernel = 'rbf', class_weight='balanced')
    
    def classify_fold_accuracy(self, X, y):
        return np.mean(cross_val_score(self.classifier, X, y, cv=5))



     

