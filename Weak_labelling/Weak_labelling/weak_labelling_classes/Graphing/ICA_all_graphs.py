import numpy as np
import sys, os
import matplotlib.pyplot as plt
import pickle
import mne
from pandas.tseries import frequencies
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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
        
        fig.savefig(os.path.join(self.current_dir, f'csp patterns {title}.png'),format = 'png')
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
            
                
        fig.savefig(os.path.join(self.current_dir, f'Freq_topoplot_{title}.png'),format = 'png')
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
        
    ######-------------------------------------------------------------------------------------------
    #3D PLOTS
        
    def plot_3D_mean(self, comp_topo_array, inst):
        
        m = self.reshape_arr(np.mean(np.stack(comp_topo_array, axis = 0), axis = 0))

        fig = plt.figure()
        
        ax = fig.add_subplot(1,1,1, projection='3d')
        self.plot_surf_ax(m, ax)
        
        ax.set_title(f'Instruction {inst} mean comp similarity')
        
        plt.savefig(os.path.join(self.current_dir, f'mean comp similarity_{inst}.pdf'))
        pickle.dump(fig, open(os.path.join(self.current_dir, f'mean comp similarity_{inst}'), 'wb'))
        plt.close()
        
    
        
    def reshape_arr(self,arr):
        return arr[1:].reshape((3,5))
    
    def plot_surf_ax(self,arr,ax):
    
        
        x,y =(np.arange(5),np.arange(3))
        X, Y = np.meshgrid(x,y)

        ax.plot_surface(X, Y, arr, ccount = 50,cmap = cm.viridis, antialiased = True)
        
    def plot_top_n(self, comp_topo_array, n, inst):
        
        best_comps = self.get_top_comps(comp_topo_array, n, inst)

        fig = plt.figure(figsize = (2*n,5))
        for i in range(len(best_comps)):
            ax = fig.add_subplot(1,n+1,i+1, projection='3d')
            self.plot_surf_ax(self.reshape_arr(comp_topo_array[i]), ax)

        fig.suptitle(f'Intruction {inst} top {n} comps in similarity')
        
        plt.savefig(os.path.join(self.current_dir, f'{n}_highest_comp_similarity_{inst}.pdf'))
        pickle.dump(fig, open(os.path.join(self.current_dir, f'{n}_highest_comp_similarity_{inst}'), 'wb'))
        plt.close()
        
    
    def get_top_comps(self, comp_topo_array, n, inst):
        
        example = self.get_comparison_BCI_TVR(inst)
        return self.get_best_comparisons(comp_topo_array,example,n)
        
    def get_comparison_BCI_TVR(self,inst):
        
        ints_dict = {
            '1': 'left',
            '2': 'right',
            '3': 'forward',
            'forward': 3,
            'right': 2,
            'left': 1
            }        
        
        if type(inst).__name__ == 'str':
            inst = ints_dict[inst]

        layout = []
        if inst == 1:
            layout = [[0,0,0,0,0],
                      [0,0,0,1,0],
                      [0,0,0,0,0]]

        elif inst == 3:
            layout = [[0,0,0,0,0],
                      [0,0,1,0,0],
                      [0,0,0,0,0]]

        elif inst == 2:
            layout = [[0,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,0,0,0]]

        return np.concatenate(([0], np.array(layout).flatten()))
    
    def get_best_comparisons(self, comps, example, n_comps):
        vals = []
        
        for i in comps:
            vals.append(self.compare_comps(example,i))
            
        best = []
        for i in range(n_comps):
            max_indx = np.argmax(vals)
            best.append(comps[max_indx])
            comps.pop(max_indx)
            vals.pop(max_indx)   
        return best
            
    def compare_comps(self,example,arr):
        #return np.abs(np.dot(example,arr)) * np.sum(np.multiply(example,arr))
        return np.dot(example, arr)/(np.linalg.norm(example)*np.linalg.norm(arr))