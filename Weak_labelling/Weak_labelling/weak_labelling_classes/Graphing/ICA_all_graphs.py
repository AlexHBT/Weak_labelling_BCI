import numpy as np
import sys, os
import matplotlib.pyplot as plt
import pickle


class ica_all_graphs(object):

    graph_dir = 'D:/Weak ICA graphs/'
    current_dir = None

    def __init__(self):
        pass
    
    def create_dir(self, name, session):
        dir_path = os.path.join(os.path.join(self.graph_dir,name),str(session))
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path,exist_ok=True)
        self.current_dir = os.path.abspath(dir_path)
        return self
        
    def plot_scatter_std(self, bags, labels = None):
        fig = plt.figure(figsize=(10,10))
        ax = fig.subplots(1)
        plots = []
        for b in bags:
           plots.append(ax.scatter(np.stack(b,axis = 0)[:,0],
                                   np.stack(b,axis = 0)[:,1]))
    
        if labels != None:
            ax.legend(plots,labels)
            ax.set_title(f"{','.join(labels)} scatter plot")
            
        fig.savefig(os.path.join(self.current_dir, 'scatter_plot.png'))
        pickle.dump(fig, open(os.path.join(self.current_dir,'scatter_plot.fig.pickle'), 'wb'))
        plt.close()    
        
    def plot_scatter_with_mean(self, bags, labels = None):
        fig = plt.figure(figsize=(10,10))
        ax = fig.subplots(1)
                

        plots = []
        for b in bags:
           plots.append(ax.scatter(np.stack(b,axis = 0)[:,0],
                                   np.stack(b,axis = 0)[:,1]))
           
        for i in range(len(bags)):
            m = self.calculate_mean(bags[i])
            plots.append(ax.scatter(m[0],m[1], marker = '^'))
            if labels != None:
                labels.append(f'{labels[i]} mean')
        if labels != None:
            ax.legend(plots,labels)
            #ax.set_title(f"{','.join(labels)} scatter plot")

        fig.savefig(os.path.join(self.current_dir, 'mean_scatter_plot.png'))
        pickle.dump(fig, open(os.path.join(self.current_dir,'mean_scatter_plot.fig.pickle'), 'wb'))
        plt.close()

    def plot_scatter(self, bags, labels = None, means = False):
        if means:
            self.plot_scatter_with_mean(bags, labels)
        else:
            self.plot_scatter_std(bags, labels)

        
        
            
    def calculate_mean(self, bag):
        return np.stack(np.mean(bag, axis = 0),axis = 0)        