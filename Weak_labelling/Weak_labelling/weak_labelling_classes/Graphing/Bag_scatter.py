from matplotlib import pyplot as plt

class Bag_scatter():
    
    fig = None
    ax = None
    
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.subplots(1,1)
    
    def plot_bag(self,bag ,marker = None):
        
        for i in bag:
            self.ax.scatter(i[:,0], i[:1], marker = marker)
        




