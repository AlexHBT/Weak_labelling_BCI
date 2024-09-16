import numpy as np
import scipy as sci
from ..bag_classes.bag import Bag
from ..bag_classes.Instruction import Instruction
from copy import copy
from ..Filters.Bag_filters import Bag_filters
from ..embedding_models.SSFFT import ssfft
from ..embedding_models.PCA import Pca
from sklearn.svm import SVC
from ..Comparable_methods.BCI_TVR import bci_tvr
import copy
class ICA_inner_2():
    
    ints_dict = {
        '1': 'left',
        '2': 'right',
        '3': 'forward',
        'forward': 3,
        'right': 2,
        'left': 1
    }
    
    def get_data_columns(self):
        return ['Method accuracy','Without method accuracy',
                'seperation_score', 
                'pre bag 1 size', 'pre bag 2 size',
                'post bag 1 size', 'post bag 2 size']
    

    def __init__(self):
        pass
    
    def test_2_classes_all(self, inst1, inst2):
        comp_method = bci_tvr()       

        bag1 = self.combine_bags(inst1.get_bags()).get_bag()
        bag2 = self.combine_bags(inst2.get_bags()).get_bag()
        orgin_acc = comp_method.process_and_classify([bag1, bag2])
        filt_bag1 = self.filter_bag(bag1)
        filt_bag2 = self.filter_bag(bag2)
        comps1 = self.convert_bag(filt_bag1,self.ints_dict[inst1.get_name().lower()])
        comps2 = self.convert_bag(filt_bag2,self.ints_dict[inst2.get_name().lower()])
        index1, index2, sep_score = self.get_kept_indexes2(comps1,comps2)
        bpl1 = len(bag1)
        bpl2 = len(bag2)
        bag1 = self.retrive_index_data(bag1, index1)
        bag2 = self.retrive_index_data(bag2, index2)
        processed_accuracy = comp_method.process_and_classify([bag1, bag2])
        return [processed_accuracy, orgin_acc, sep_score, bpl1, bpl2, len(index1), len(index2)]
        
    
        

    def get_comparison_BCI_TVR(self,inst):
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
    
    def compare_values(self,v1,v2):
        v1 = v1* np.max(v2)
        #v2 = v2/np.linalg.norm(v2)
        #print(v1)
        #print(v2)
        return np.inner(v1,v2)
    
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
    
    def get_bag_mean(self,bag):
        return np.mean(np.stack(bag, axis = 0), axis = 0)  

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
    
    def get_components(self,example, instruction, n_comps = 3):
        sources, mix = self.ICA_data(example)
        Im = np.linalg.inv(mix).real

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
    
        comps = example[:, max_indexes]
        return comps
    
    def convert_bag(self,bag:[np.ndarray], inst):
        
        new_comps = []
        for i in bag:
            new_comps.append(self.get_components(i, inst))

        return new_comps
    
    def filter_bag(self, bag):
        bf = Bag_filters()
        return bf.filter_bag(bag)
    
    def fft_bag(self,bag):
        s = ssfft()
        return s.fft_compress(bag)
    
    def stack_flatten(self, bag):
        return self.stack_data(self.flatten_data(bag))
    
    def PCA_data(self, bags:[[np.ndarray],[np.ndarray]]):
        p = Pca()
        return p.PCA_bags(bags)
        
    def cos_sim(self,a,b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
    def get_kept_indexes2(self, bag1, bag2):
        X, y = self.PCA_data([bag1,bag2])
        bag1, bag2 = self.split_bag_2(X,y)
        m1 = self.get_bag_mean(bag1)
        m2 = self.get_bag_mean(bag2)
        sep1 = self.get_sep(m1,m2)
        sep2 = self.get_sep(m2,m1)
        bag1, index1 = self.calc_kept_values(sep1, bag1)
        bag2, index2 = self.calc_kept_values(sep2, bag2)
        sep_score = self.test_seperation([bag1,bag2])
        return index1, index2, sep_score

    def get_sep(self, pos, neg):
        return pos-neg
    
    def calc_kept_values(self,sep,bag):
        kept_values = []
        indexes = []
        count = 0
        for i in bag:
            if self.cos_sim(sep, i) > 0.7:
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
    
    
        