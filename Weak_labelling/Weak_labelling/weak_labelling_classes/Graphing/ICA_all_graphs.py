import numpy as np
import sys, os
import matplotlib.pyplot as plt
import pickle
import mne
from pandas.tseries import frequencies
import matplotlib as mpl


class ica_all_graphs(object):

    #graph_dir = 'D:/Weak ICA graphs/'
    graph_dir = 'E:/Alex/Weak labelling test/Graphs'
    current_dir = None

    def __init__(self):
        mne.set_log_level('CRITICAL')
    
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
    

    def plot_csp_filters(self, bags, title = ''):
        X,y = self.convert_to_ml_data(bags)
        c = mne.decoding.CSP()
        c.fit_transform(X,y)
        info = mne.create_info(['Fz','FC3','FC1','FCz','FC2','FC4',
                                'C3','C1','Cz','C2','C4',
                                'CP3','CP1','CPz','CP2','CP4'],512,'eeg')
        info.set_montage(mne.channels.make_standard_montage('standard_1020'))
        
        fig = c.plot_patterns(info,ch_type="eeg", units="Patterns (AU)", size=1.5, show = False)
        
        fig.savefig(os.path.join(self.current_dir, f'csp patterns {title}.png'))
        pickle.dump(fig, open(os.path.join(self.current_dir,f'csp patterns{title}.fig.pickle'), 'wb'))
        plt.close()
        

    def convert_to_ml_data(self,bags):
        y = []
        total_data = []
        for b in range(len(bags)):
            y.append(np.zeros(len(bags[b])) + b)
            for i in bags[b]:
                total_data.append(i)
        total_data = np.stack(total_data, axis = 0)
        return np.swapaxes(total_data,1,2), np.concatenate(y)
    
    def freq_topoplot(self, bags:list, title = ''):
        frequencies = [8,10,12,14,16,18]
        
        fig = plt.figure()
        axs = fig.subplots(len(bags),bags[0][0].shape[0])
        for i in range (len(bags)):
            mean_freq = np.mean(np.stack(bags[i], axis = 0), axis = 0)
            for j in range(mean_freq.shape[0]):
                ax = axs[i,j]
                self.plot_freq_topoplot(mean_freq[i,:],ax)
                ax.set_title(f'{frequencies[j]}')
                if j == 0:
                    #ax.axis("on")
                    #ax.get_xaxis().set_visible(False)
                    ax.set_ylabel(f'Bag {i}')
            
                
        fig.savefig(os.path.join(self.current_dir, f'Freq_topoplot_{title}.png'))
        pickle.dump(fig, open(os.path.join(self.current_dir,f'Freq_topoplot_{title}.fig.pickle'), 'wb'))
        plt.close()
        
        
    def plot_freq_topoplot(self,freq, ax):
        layout = [[-1,-1,0,-1,-1],
                  [1,2,3,4,5],
                  [6,7,8,9,10],
                  [11,12,13,14,15]]

        plot = layout[:]

        for i in range(len(layout)):
            for j in range(len(layout[0])):

                if layout[i][j] == -1:
                    plot[i][j] = np.mean(freq)
                else:
                    plot[i][j] = freq[layout[i][j]]

        im = ax.imshow(plot, cmap = 'plasma', interpolation = 'spline16')
        patch = mpl.patches.Circle((2, 1.5), radius=2, transform=ax.transData)
        im.set_clip_path(patch)
        ax.axis("off")