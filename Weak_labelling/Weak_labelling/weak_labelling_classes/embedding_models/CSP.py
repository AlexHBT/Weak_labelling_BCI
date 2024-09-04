from mne.decoding import CSP
import copy
import numpy as np
import mne

class csp():
    
    transformer = None
    
    def __init__(self, components = 5):
        
        mne.set_log_level('WARNING')
        
        self.transformer = CSP(n_components = components,transform_into= 'csp_space')
        
    
    def fit(self, insts):
        
        X, y = self.get_combined_data(insts)
        
        self.transformer = self.transformer.fit(X,y)
        
        for i in insts:
            for b in range(i.get_bags_length()):
                temp_bag = []
                for e in range(i.get_bag(b).length()):
                    temp_bag.append(np.squeeze(
                        self.transformer.transform(
                        i.get_bag(b).get_example(e).T[np.newaxis,...])))
                
                i.get_bag(b).set_bag(temp_bag)
            



    def get_combined_data(self,insts):
        
        y = []
        X = []
        
        for i in range(len(insts)):
            for j in range(insts[i].get_bags_length()):
                temp_vals = insts[i].get_bag(j).get_bag()
                X.extend(temp_vals)
                y.append(np.ones(len(temp_vals))*i)
            
        X = np.swapaxes(np.stack(X, axis = 0),1,2)
        y = np.concatenate(y)
        
        return X, y
    
     