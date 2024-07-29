from mne.decoding import CSP
class csp():
    
    transformer = None
    
    def __init__(self, components):
        
        self.transformer = CSP(n_components = components)
        
    
    def fit_and_return(self, X, y):
        return self.transformer.fit_transform(X,y)




