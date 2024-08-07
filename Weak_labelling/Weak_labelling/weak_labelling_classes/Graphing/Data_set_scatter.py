from matplotlib import pyplot as plt
import numpy as np
import sklearn.decomposition
class data_set_scatter():
    
    ax = None
    fig = None
       
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.subplots(1,1)
    
    def plot_inst(self, inst):
        pass
    
    def plot_dat_set(self, X, y):
        pca = sklearn.decomposition.PCA(n_components=2)
        X = pca.fit_transform(X)
        
        self.ax.scatter(X[:,0], X[:,1], c = y)
        self.ax.set_title('Post processing data principle components')
        
        plt.show(block = True)
        




