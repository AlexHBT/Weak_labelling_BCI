from sklearn.decomposition import PCA
import numpy as np
class Pca():
    
    Transform = None
    
    def __init__(self, n_components = 2):
        self.Transform = PCA(n_components = n_components)
        
    def fit(self,X):
        return self.Transform.fit_transform(X)
        
    def PCA_bags(self,bags, ncomps =2):
        y = []
        total_data = []
        for b in range(len(bags)):
            y.append(np.zeros(len(bags[b])) + b)
            for i in bags[b]:
                total_data.append(i.reshape(1,-1))
        total_data = np.concatenate(total_data, axis = 0)
        pca = PCA(ncomps)
        total_data = pca.fit_transform(total_data)
        return total_data, np.concatenate(y)
    
    
