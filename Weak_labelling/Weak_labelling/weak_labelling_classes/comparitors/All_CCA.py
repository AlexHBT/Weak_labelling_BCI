
from calendar import c

from six import b
from ..Comparable_methods.BCI_TVR import bci_tvr
from ..bag_classes.bag import Bag
from ..bag_classes.Instruction import Instruction
import os
import numpy as np
from sklearn.cross_decomposition import CCA
from ..Graphing.ICA_all_graphs import ica_all_graphs
from ..Filters.Bag_filters import Bag_filters
class all_cca():
    

    graph = None
    save_inst = None


    def __init__(self, session, name):
        self.session = f'session {session}'
        self.name = name
        self.save_inst = 0

    def test_2_classes_all(self, inst1, inst2):
        self.graph = ica_all_graphs().create_dir(self.name, self.session)
        
        comp_method = bci_tvr(self.graph)
        #comp_method = csp_classifier()

        bag1 = self.combine_bags(inst1.get_bags()).get_bag()
        bag2 = self.combine_bags(inst2.get_bags()).get_bag()

        bag1 = self.filter_bag(bag1)
        bag2 = self.filter_bag(bag2)
                

        b1 = self.CCA_bag(self.select_channel_bag(bag1, 9))
        b2 = self.CCA_bag(self.select_channel_bag(bag2, 7))
        
        self.save_embeddings([b1,b2])
        

    def select_channel_bag(self,bag, channel):
        
        new_bag = []
        
        for inst in bag:
            new_bag.append(inst[:,channel]) 
            
        return new_bag


    def CCA_bag(self, bag):
        
        bag = np.stack(bag, axis = 0)

        comps = []        
        comp = self.get_comp()
        
        for i in range(bag.shape[0]):
            comps.append(comp)
            
        comps = np.stack(comps)
            
        embedding = self.perform_CCA(bag,comps)
        
        return self.flatten_data(np.stack(embedding, axis = 2))
            
        


    def get_comp(self):
        frequencies = list(range(41))
        frequencies.pop(0)
        values = [8,9,10,11,12,13]
        for i in values:
            frequencies.remove(i)
        comp = self.genorate_comp(np.array(frequencies))
        return comp

    def genorate_comp(self,frequencies):
    
        inp = np.arange(0,2*np.pi, (2*np.pi)/512)

        waves = []
        for f in frequencies:
            waves.append(np.cos(inp*f))
        comps = np.sum(np.stack(waves, axis = 0), axis = 0)/frequencies.shape[0]

        return comps
        
    def perform_CCA(self, examples, targets):
        return CCA().fit_transform(examples,targets)
        
        
    def filter_bag(self, bag):
        bf = Bag_filters()
        return bf.broad_bag(bf.sl_bag(bag))
        #return bf.filter_bag(bag)
        
    
    def save_embeddings(self,inst:[]):
         self.save_inst += 1
         file = 'D:/Weak_labelling ICA embeddings/'
         X = []
         y = []
         for i in range(len(inst)):
             y.append(np.zeros(len(inst[i]))+i)
             X.append(np.stack(inst[i], axis = 0))
             
         X = np.concatenate(X, axis = 0)
         y = np.concatenate(y, axis = 0)
         data = np.concatenate((X,y[:,np.newaxis]), axis = 1)
         np.save(f'{file}/{self.name}_{self.session.replace(" ","_")}_{self.save_inst}.npy', data)
         #if self.save_inst>1:
         #raise Exception("Skipping classification")

    def combine_bags(self,bags):
        

        new_bag = Bag()
        for b in bags:
            new_bag.extend_bag(b.get_bag())
        return new_bag
    
    def flatten_data(self, data:np.ndarray):
        split_data = np.split(data, data.shape[0], axis= 0)
        
        new_data = []
        for i in split_data:
             new_data.append(np.squeeze(i.flatten()))
            
        return np.stack(new_data, axis = 0)