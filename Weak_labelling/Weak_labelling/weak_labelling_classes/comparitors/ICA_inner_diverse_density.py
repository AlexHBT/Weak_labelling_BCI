
import os
import graphlib
import numpy as np
import scipy as sci
from ..bag_classes.bag import Bag
from ..bag_classes.Instruction import Instruction
from copy import copy
from ..Filters.Bag_filters import Bag_filters
from ..embedding_models.SSFFT import ssfft
from ..embedding_models.PCA import Pca
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.ensemble import IsolationForest
from ..Metrics import Diverse_Density

import copy
from ..Metrics.ICA_metrics import ica_metrics
from ..Graphing.ICA_all_graphs import ica_all_graphs
from .ICA_Inner_measures import sim_measures

from ..Filters.outlier import outlier_dection

## comparable method
from ..Comparable_methods.BCI_TVR import bci_tvr
from ..Comparable_methods.CSP_LDA import csp_classifier
from ..Comparable_methods.EEGnet import eegnet


class ICA_inner_diverse_desnity():
    
    ints_dict = {
        '1': 'left',
        '2': 'right',
        '3': 'forward',
        'forward': 3,
        'right': 2,
        'left': 1
    }
    
    session = None
    
    name = None
    
    graph = None
    
    comps = None
    comp_treshold = 0.5
    
    def get_data_columns(self):
        return ['Method accuracy','Without method accuracy',
                'seperation_score1','seperation_score2', 
                'pre bag 1 size', 'pre bag 2 size',
                'post bag 1 size', 'post bag 2 size',
                'dloss']
    

    def __init__(self, session, name):
        self.session = f'session {session}'
        self.name = name
        
        self.comps = []
        
    def retrieve_indexes(self,pos_bags, neg_bags, pos_inst):
        
        pos_inst_comps = self.convert_inst(pos_bags,self.ints_dict[pos_inst.get_name().lower()], is_pos = True)
        neg_inst_comps = self.convert_inst(neg_bags,self.ints_dict[pos_inst.get_name().lower()], is_pos = False)
        
        pos_indexes, sep_score = self.get_kept_indexes2(pos_inst_comps,neg_inst_comps, pos_inst.get_name().lower())
        
        return pos_indexes, sep_score
        
    def test_2_classes(self, inst1, inst2):
        
        self.graph = ica_all_graphs().create_dir(self.name, self.session)
        #comp_method = bci_tvr(self.graph)
        comp_method = csp_classifier()
        
        bag1 = self.combine_bags(inst1.get_bags()).get_bag()
        bag2 = self.combine_bags(inst2.get_bags()).get_bag()
       
        orgin_acc = comp_method.process_and_classify([bag1, bag2])

        self.plot_csp_patterns([bag1, bag2], 'standard')        

        filt_inst_1 = []
        filt_inst_2 = []
        
        for i in range(inst1.get_bags_length):
            filt_inst_1.append(self.filter_bag(inst1.get_bag(i, copy = True).get_bag()))
            
        for i in range(inst2.get_bags_length):
            filt_inst_2.append(self.filter_bag(inst2.get_bag(i, copy = True).get_bag()))

        
        
        index1, max_dd_1 = self.retrieve_indexes(filt_inst_1, filt_inst_2, inst1)
        index2, max_dd_2 = self.retrieve_indexes(filt_inst_2, filt_inst_1, inst2)
        
        
        bpl1 = len(bag1)
        bpl2 = len(bag2)
        bag1 = self.retrive_index_data(bag1, index1)
        bag2 = self.retrive_index_data(bag2, index2)
        
        #self.plot_csp_patterns([bag1, bag2], 'after')  
        processed_accuracy = comp_method.process_and_classify([bag1, bag2])
        #dloss = self.get_dloss([bpl1, bpl2], [len(index1), len(index2)])
        self.plot_similarit_graphs()
        return [processed_accuracy, orgin_acc, max_dd_1, max_dd_2, bpl1, bpl2, len(index1), len(index2)]#dloss]
        
    
    def normalize_im(self, im):
        return normalize(np.nan_to_num(im), axis = 1)
        #return im
    def get_comparison_BCI_TVR(self,inst):
        layout = []
        if inst == 1:
            layout = [[0,0,0,0,0],
                      [0,0,0,1-0.2,0],
                      [0,0,0,0,0]]

        elif inst == 3:
            layout = [[0,0,0,0,0],
                      [0,0,1-0.2,0,0],
                      [0,0,0,0,0]]

        elif inst == 2:
            layout = [[0,0,0,0,0],
                      [0,1-0.2,0,0,0],
                      [0,0,0,0,0]]

        return np.concatenate(([0], np.array(layout).flatten()))+0.2 
        #return self.get_comparison_rand(inst)
    
    def get_comparison_rand(self, inst):
        
         layout = np.random.rand(3,5)
         return np.concatenate(([0], np.array(layout).flatten())) 
    
    
    def compare_values(self,v1,v2):
            
        sm = sim_measures() 
        #v1 = v1* np.max(v2)
        #v2 = v2/np.linalg.norm(v2)
        #print(v1)
        #print(v2)
        #return np.inner(v1,v2)
        #return self.cos_sim(v1,v2)
        return sm.cos_sim(v1,v2)
    
    def whitten(self,epoch):
        E, D = np.linalg.eig(np.cov(epoch.T))
        white_epoch =  np.matmul(sci.linalg.sqrtm(np.linalg.pinv(D)) * E.T,  epoch.T)
        return white_epoch.T, E, D
    
    def get_mixing(self,X, E, D):
        #print(D.shape)
        V, s, u = np.linalg.svd(np.matmul(np.multiply(
                        np.sum(np.multiply(X,X), axis = 0), X).T, X))
        #print (V.shape)
        W = V * sci.linalg.sqrtm(np.linalg.pinv(D)) * E.T
        return W
    
    def ICA_data(self,epoch):
        white, E, D = self.whitten(epoch)
        #print (white.shape)
        mixing = self.get_mixing(white, E, D)
        sources = np.matmul(epoch, mixing)
        return sources, mixing
    
    def flatten_inst(self, inst):
        new_inst = []
        for b in inst:
            new_inst.append(self.flatten_data(b))
            
        return new_inst
            
    def flatten_data(self,bag):
        new_values = []

        for i in bag:
            new_values.append(i.flatten())
        return new_values

    def stack_data(self,bag):
        return np.stack(bag, axis = 0)

    def concatenate_data(self,bag):
        return np.concatenate(bag, axis = 0)

    def combine_bags(self,bags):
        new_bag = Bag()
        for b in bags:
            new_bag.extend_bag(b.get_bag())
        return new_bag


    def split_bag_2(self,X,y):
        bag_1 = []
        bag_2 = []

        for i in range(len(y)):
            if y[i] == 0:
                bag_1.append(X[i,:])
            elif y[i] == 1:
                bag_2.append(X[i,:])   
            else:
                pass
    
        return bag_1, bag_2
    
    def get_components(self,example, instruction, n_comps = 3, is_pos = True):
       
        sources, mix = self.ICA_data(example)
        Im = self.normalize_im(np.linalg.inv(mix).real)
        
        
        values = []
        perfect_example = self.get_comparison_BCI_TVR(instruction)

        for i in range(Im.shape[0]):
            values.append(abs(self.compare_values(perfect_example, Im[:,i])))

    
    
        #prints out values
        order_list = copy.deepcopy(values)
        order_list.sort()
        #print(order_list[:3])
    
        max_indexes = []
        for i in range(n_comps):
            max_indexes.append(np.argmax(values))
            values.pop(max_indexes[-1])

        #plots the main value
        #plot_comps(Im[:,max_indexes],instruction)
    
        comps = sources.real[:, max_indexes]
        self.save_array(Im)
        self.save_array_inst(Im[:,max_indexes], instruction)
        if is_pos:
            self.comps[-1].extend(self.split_array(Im[:,max_indexes],axis = 1))
         
            
        
        return comps
    
    def convert_inst(self, inst_array:[[np.ndarray]],inst: Instruction, is_pos:bool = True):
        
        new_inst = []

        for i in inst:
            new_inst.append(self.convert_bag(i, inst, is_pos))
            
        return new_inst
    
    def convert_bag(self,bag:[np.ndarray], inst, is_pos = True):
        if is_pos: 
            self.comps.append([])
        new_comps = []
        for i in bag:
            new_comps.append(self.get_components(i, inst,is_pos=is_pos))

        return new_comps
    
    def filter_bag(self, bag):
        bf = Bag_filters()
        return bf.broad_bag(bf.sl_bag(bag))
        #return bf.filter_bag(bag)
    
    def fft_inst(self, inst):
        new_inst = []
        
        for b in inst:
            new_inst.append(self.fft_bag(b))
        
        return new_inst
    
    def fft_bag(self,bag):
        s = ssfft()
        return s.fft_compress(bag)
    
    def stack_flatten(self, bag):
        return self.stack_data(self.flatten_data(bag))
    
    def PCA_data(self, bags:[[np.ndarray],[np.ndarray]], ncomps = 6):
        p = Pca()
        return p.PCA_bags(bags, ncomps = ncomps) # change pca comps
        
    def cos_sim(self,a,b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
    
    
    def get_kept_indexes2(self, pos_inst, neg_inst, pos_name = ''):
        

        pos_inst = self.flatten_inst(self.fft_inst(pos_inst))
        neg_inst = self.flatten_inst(self.fft_inst(neg_inst))
        
        od = outlier_dection()
        
        #pos_bag = od.Isolation_forrest(pos_bag)
        #neg_bag = od.Isolation_forrest(neg_bag)
        
        
        
        #X, y = self.PCA_data([pos_bag,neg_bag])
        #X,y = self.convert_to_ml_data([bag1,bag2])
        #pos_bag, neg_bag = self.split_bag_2(X,y)
        

        #bag1 = self.flatten_data(bag1)
        #bag2 = self.flatten_data(bag2)
        
        dd_scores = self.get_diverse_density(pos_inst, neg_inst)


        pos_all = self.combine_bags(pos_inst).get_bag()
        neg_all = self.combine_bags(neg_inst).get_bag()
        dd_all = self.combine_bags(dd_scores).get_bag()
        
        max_dd = pos_all[np.argmax(dd_all)]
        
        pos_bag, index1 = self.calc_kept_values(max_dd, pos_all)
        #bag2, index2 = self.calc_kept_values(sep2, bag2)
        

        
        return index1, max_dd

    
    
    def calc_kept_values(self,max_dd,bag):
        kept_values = []
        indexes = []
        count = 0
        for i in bag:
            if self.euclid_dist(max_dd, bag) > self.comp_treshold: # Threshold value
                kept_values.append(i)
                indexes.append(count)
            count+= 1
            
        return kept_values, indexes
    
    def convert_to_ml_data(self,bags):
        y = []
        total_data = []
        for b in range(len(bags)):
            y.append(np.zeros(len(bags[b])) + b)
            for i in bags[b]:
                total_data.append(i.reshape(1,-1))
        total_data = np.concatenate(total_data, axis = 0)
        return total_data, np.concatenate(y)
    
    def linear_classify(self,X,y):
        cl = SVC(kernel = 'linear').fit(X,y)
        return cl.score(X,y)
    
    def test_seperation(self,bags):
        X,y = self.convert_to_ml_data(bags)
        return self.linear_classify(X,y)
    
    def retrive_index_data(self,bag, indexes):
        kept_epochs = []
        #print(f'length of bag {len(bag)}')
        for i in indexes:
            #print(i)
            kept_epochs.append(bag[i])

        return kept_epochs
    
    
    def get_dloss(self,bag_sizes,r_sizes):
        m = ica_metrics()
        return m.data_loss(bag_sizes,r_sizes)
    
    def plot_csp_patterns(self, bags, title):
        self.graph.plot_csp_filters(bags, title)
        
    def save_array(self,inverse_mix):
        file = 'D:/Weak ICA components/arrays.npy'
            
        if os.path.isfile(file):
            arr = np.load(file)
            arr = np.concatenate((arr,inverse_mix.T), axis = 0)
            np.save(file, arr)
            
        else:
            np.save(file, inverse_mix.T)
            
    def save_array_inst(self,inverse_mix,inst):
        file = f'D:/Weak ICA components/{str(inst)}_comp_arrays.npy'
            
        if os.path.isfile(file):
            arr = np.load(file)
            arr = np.concatenate((arr,inverse_mix.T[0,:][np.newaxis,...]), axis = 0)
            np.save(file, arr)
            
        else:
            np.save(file, inverse_mix.T[0,:][np.newaxis,...])
            

    def plot_similarit_graphs(self, inst_names = None):
        inst_names = ['left','right']
        for i in range(len(self.comps)):
            self.graph.plot_3D_mean(self.comps[i], inst_names[i])
            self.graph.plot_top_n(self.comps[i],10,inst_names[i])
            
    def split_array(self, arr: np.ndarray, axis:int ):
        s = np.split(arr, arr.shape[axis], axis = axis)
        
        for i in range(len(s)):
            s[i] = np.squeeze(s[i])
            
        return s
    
    def anomally_remove(self, bag):
        b = np.stack(bag,axis = 0)
        anoms = IsolationForest().fit_predict(b).tolist()
        
        new_bag = []
        for i in range(len(anoms)):
            if anoms[i] == 1:
                new_bag.append(bag[i])
                
        return new_bag
            
    def plot_2D_scatter(self, bags, class_names ,means = False, name = ''):
        X,y = self.PCA_data(bags,2)
        pos_bag, neg_bag = self.split_bag_2(X,y)
        self.graph.plot_scatter([pos_bag,neg_bag], class_names, means = means, name = name)


    def calculate_diverse_density(self, positive_bags, negative_bags, x):
        dd = Diverse_Density.diverse_denisty()
        
        score = dd.test_position(positive_bags, negative_bags,x)
        return score
        

    def test_bag_positions(self, pos_bag, positive_bags,negative_bags):
        bag_values = []

        for x in pos_bag:
            bag_values.append(self.calculate_diverse_density(positive_bags,negative_bags, x))
            
        return bag_values
            
            
    def get_diverse_density(self, positive_bags, negative_bags):
        dd_per_bag = []
        for b in positive_bags:
            dd_per_bag.append(self.test_bag_positions(b, positive_bags, negative_bags))
            
        return dd_per_bag
    
    def euclid_dist(self, a,b):
        return np.linalg.norm(a-b)