import pandas as pd
import numpy as np
from ..data_loaders.CSV_loader import CSV_loader


from ..Filters.Butterworth import butterworth
from ..embedding_models.SSFFT import ssfft
from ..comparitors.Mean_bag_comparitor import Mean_bag
from ..classifiers.SVM import SVM 
#from ..Graphing.Bag_scatter import Bag_scatter
from ..embedding_models.PCA import Pca
#from ..Graphing.Data_set_scatter import data_set_scatter
from ..comparitors.CPS_mean_bag import CSP_mean_bag
from ..comparitors import SSFFT_mean_bag
from ..comparitors import ICA_inner_product
from ..embedding_models import CSP

from ..classifiers.MLP import Mlp

from ..comparitors.ICA_inner import ICA_inner_2
from ..comparitors.ICA_inner_diverse_density import ICA_inner_diverse_desnity


class participant_test(object):
    
    classifier = None
    instructions = None
    test_instructions = None
    
    
        
    def start_2(self, train_files: [str], name:str, session:int):
        
        self.instructions = self.load_file_data(train_files)
        self.instructions = self.instructions[1:]
        self.instructions.pop(1)
        
        #test = ICA_inner_diverse_desnity(session,name)
        #return test.test_2_classes(self.instructions[0],self.instructions[1])

        test = ICA_inner_2(session,name)
        return test.test_2_classes_all(self.instructions[0],self.instructions[1])
        
    def load_file_data(self, files):
        csvl = CSV_loader()
        
        instruction_sets = []
        
        for f in files:
            instruction_sets.append(csvl.load_bag_CSV(f))
            
        instructions = instruction_sets[0]
        instruction_sets.pop(0)
            
        for in_set in instruction_sets:
            for j in in_set:
                
                for a in instructions:
                    if j.get_name() == a.get_name():
                        a.add_bags(j.get_bags())
                        
            
            
                
        return instructions
    
            
    def filter_pipe_line(self):
        b = butterworth()
        s = ssfft()
        pipeline = [b.filter_signal]
        
        for i in self.instructions:
            for p in pipeline:
                #print(f'Filtering {i.get_name()}')
                i.parse_callable(p)
            #print('debug') 
    
    
    
            
    def print_lengths(self):
        print(f'Number of instructions:{len(self.instructions)}\n')
        
        print(f'instructions:{len(self.instructions)}\n')
        for i in self.instructions:
            print(f'Instruction 1 : {i.get_name()}: bags: {i.get_bags_length()}')
            print(f'Instruction 1 first bag length: {i.get_bag(0).length()}\n')
            
       
    def classify_data(self):
        
        self.classifier = SVM()

        #self.classifier = Mlp()
        
        X, y = self.get_ML_data()
        
        #data_set_scatter().plot_dat_set(X,y)
        
        #pca_transformer = Pca(5)
        #X = pca_transformer.fit(X)
        
        return self.classifier.classify_fold_accuracy(X,y)
            
        
    def get_ML_data(self):
        label_list = []
        y_list = []
        X = []
        
        for i in range(len(self.instructions)):
            data = self.instructions[i].get_vecorized_data()
            y_list.append(np.zeros(len(data))+i)
            label_list.append(self.instructions[i].get_name())
            X.append(data)
            
        X = np.concatenate(X, axis = 0)
        y = np.concatenate(y_list, axis = 0)
        
        return X, y
    
    def get_test_ML_data(self):
        label_list = []
        y_list = []
        X = []
        
        for i in range(len(self.test_instructions)):
            data = self.test_instructions[i].get_vecorized_data()
            y_list.append(np.zeros(len(data))+i)
            label_list.append(self.test_instructions[i].get_name())
            X.append(data)
            
        X = np.concatenate(X, axis = 0)
        y = np.concatenate(y_list, axis = 0)
        
        return X, y
    
    def classify_test_data(self):
        
        
        
        X, y = self.get_test_ML_data()
        
        #data_set_scatter().plot_dat_set(X,y)
        
        #pca_transformer = Pca(5)
        #X = pca_transformer.fit(X)
        
        return self.classifier.evaluate(X,y)
    
    def compare(self):
        #comparitor = CSP_mean_bag(5)
        #comparitor.compare_against_all(self.instructions)
        #comparitor = SSFFT_mean_bag.FFT_mean_bag()
        comparitor = ICA_inner_product.ICA_inner()
        comparitor.compare_against_all(self.instructions)
        return self.count_instructions()
        
    
        
    def post_process(self):
        csp = CSP.csp()
        csp.fit(self.instructions)
        if self.test_instructions != None:
            csp.fit(self.test_instructions)
            

    def count_instructions(self):
        counts = []
        count = 0
        for i in self.instructions:
            count = 0
            for bag in i.get_bags():
                count += bag.length()
            counts.append(count)
            
        return counts


