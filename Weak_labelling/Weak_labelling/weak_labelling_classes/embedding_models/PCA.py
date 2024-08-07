from sklearn.decomposition import PCA
class Pca():
    
    Transform = None
    
    def __init__(self, n_components = 2):
        self.Transform = PCA(n_components = n_components)
        
    def fit(self,X):
        return self.Transform.fit(X)
        



