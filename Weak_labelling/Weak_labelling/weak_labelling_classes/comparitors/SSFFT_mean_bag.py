from mne.decoding import CSP
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition 
import datetime
import os
import scipy.fft

class FFT_mean_bag():
    
    epoch_length = None
    comps = None
    
    threshhold = None
    
    make_graphs = False
    
    instruction_type = None 
    
    plot_dir = 'D:\Weak_labelling_Graph_plots'

    def __init__(self):
        
        self.threshhold = 0.5
        self.epoch_length = 1
        
    
    def compare_against_all(self, instructions:list):
            
        values = []
        
        for i in range(len(instructions)):
            self.instruction_type = i
            pos = instructions[i]
            negs = instructions[:]
            negs.pop(i)
            values.append(self.compare_inst(pos,negs)) 
            
        self.remove_instances(instructions, values)
      
    def compare_inst(self, pos_inst, neg_insts):
        
        neg_bags = []
        
        for n in neg_insts:
            neg_bags.extend(n.get_bags())
            
        pos_bag_values = []
        for p in range(pos_inst.get_bags_length()):
            pos_bag_values.append(self.compare_one_agaist_many(pos_inst.get_bag(p),neg_bags))
            
        return pos_bag_values
        

    def compare_one_agaist_many(self, pos_bag, neg_bags):
            

        all_neg_examples = []

        for n in neg_bags:
            
            all_neg_examples.extend(n.get_bag())
          
        return self.compare_bag(pos_bag.get_bag(), all_neg_examples)
    
              
    def compare_bag(self, pos_bag, neg_bag):
        
        pos, neg = self.embed_fft(pos_bag, neg_bag)
        
        pos = self.flatten_array(pos)
        neg = self.flatten_array(neg)
        if self.make_graphs:
            self.plot_bag(pos,neg)        

        difference = np.mean(np.stack(pos, axis = 0), axis = 0) - np.mean(np.stack(neg, axis = 0), axis = 0)
        
        
        comparisons = []
        
        for i in range(len(pos)):
            comparisons.append(self.cosign_sim(pos[i],
                                               difference 
                                               ))
            
        return comparisons
        
            
    def cosign_sim(self,vec1,vec2):
        return np.abs(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    
    def inner_prod(self, vec1, vec2):
        return np.dot(vec1, vec2)
    
    def embed_fft(self, pos_bag, neg_bag):
        

        pos = []
        neg = []
        
        for i in pos_bag:
            pos.append(self.ssfft_fit(i))
        for i in neg_bag:
            neg.append(self.ssfft_fit(i))
           
        return pos, neg
        

            
    def remove_instances(self, insts, values):
        
        for i in range(len(insts)):
            for b in range(insts[i].get_bags_length()):
                
                indexes = []
                
                for v in range(len(values[i][b])):
                    if values[i][b][v] < self.threshhold:
                        indexes.append(v)
                                
                print(f'removed {len(indexes)} values, from bag {b} for {insts[i].get_name()}')
                insts[i].get_bag(b).remove_examples(indexes)
                
    def flatten_array(self,arr):
        
        temp = []
        for a in arr:
            temp.append(a.flatten())
            
        return temp 
    
    def plot_bag(self, pos, neg):
        names = ['left','right','forward']
        
        self.plot_dist(pos, neg,names[self.instruction_type])
        
        X, y = self.get_labels(pos,neg)
        
        self.plot_scatter(X,y,names[self.instruction_type])
        
        plt.close('all')
        
        
        
    def plot_scatter(self, X, y, inst_name):
        
        fig = plt.figure()
        ax = fig.subplots(1,1)
        
        X = sklearn.decomposition.PCA(2).fit_transform(X)
        
        ax.scatter(X[:,0], X[:,1], c = y)        
        ax.set_title(f'{inst_name} vs ')
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')

        time = datetime.datetime.now().strftime('%H-%M-%S')       

        fig.savefig(os.path.join(self.plot_dir,f'Scatter {inst_name} - {time}')) 
        
        
    def plot_dist(self, pos, neg, inst_name):
        
        pos = np.stack(pos, axis = 0)
        neg = np.stack(neg, axis = 0)
        
        fig = plt.figure(figsize=(5,10))
        axs = fig.subplots(pos.shape[1],1)
        
        for i in range(pos.shape[1]):
            ax = axs[i]
            
            ax.hist(neg[:,i], color = (0,0,1,0.5), label = 'Negative')
            ax.hist(pos[:,i], color = (1,0,0,1), label = 'Positive')
        ax[0].legend()
            
            
        
        axs[0].set_title(f'{inst_name} vs ')
        time = datetime.datetime.now().strftime('%H-%M-%S')     
        fig.tight_layout()
        fig.savefig(os.path.join(self.plot_dir,f'Dist {inst_name} - {time}.png'))
        

    def get_labels(self, pos, neg):
        
        pos = np.stack(pos, axis = 0)
        neg = np.stack(neg, axis = 0)
        
        pos_y = np.zeros(pos.shape[0])
        neg_y = np.ones(neg.shape[0])
        
        X = np.concatenate((pos, neg))
        y = np.concatenate((pos_y, neg_y))
        
        return X,y

   
    def ssfft_fit(self, signal):
        
        y = scipy.fft.fft(signal)
        
        
        P2 = np.absolute (y/y.shape[0])
        
        P1 = P2[1:int(y.shape[0]/2)+1]
        P1[2:-2] = 2*P1[2:-2]
        P1 = P1[int(8*self.epoch_length):int(14*self.epoch_length)] 
        return self.compress_bags(P1)
    
    def compress_bags(self, signal:np.ndarray):
        
        new_signal = []
        
        for i in range(signal.shape[1]):
            new_signal.append(np.mean(signal[:,i].reshape((2,-1), order = 'f'), axis = 0))
            
        return np.squeeze(np.stack(new_signal, axis = 1))