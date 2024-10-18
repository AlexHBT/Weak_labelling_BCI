import pandas as pd
import numpy as np
from scipy.stats import f
from weak_labelling_classes.data_loaders.CSV_loader import CSV_loader
import itertools
import os
from tkinter import filedialog
from weak_labelling_classes.Filters.Butterworth import butterworth
from weak_labelling_classes.embedding_models.SSFFT import ssfft
from weak_labelling_classes.comparitors.Mean_bag_comparitor import Mean_bag
from weak_labelling_classes.classifiers.SVM import SVM
#from weak_labelling_classes.Graphing.Bag_scatter import Bag_scatter
from weak_labelling_classes.embedding_models.PCA import Pca
#from weak_labelling_classes.Graphing.Data_set_scatter import data_set_scatter
from weak_labelling_classes.comparitors.CPS_mean_bag import CSP_mean_bag
from weak_labelling_classes.embedding_models.unembedded import unembedded

import pandas as pd

class dataset_test():
        
    def __init__(self):
        pass
        
    def start(self):
        
        files = self.get_file_list()
        
        results = []
        
        
        for f in files:
            results.append(self.test_data(self.load_file_data([f])[1:]))
            
        save_dir = 'D:/Weak_labelling_results/'
        
        for r in range(len(results)):
            
            df = pd.DataFrame(results[r])
            df.to_csv(os.path.join(os.path.abspath(save_dir),
                                   self.get_file_folder_name(files[r])))
            
    def get_file_folder_name(self, file_name):
        
        name = ""

        for i in range(3):
            split_path = os.path.split(file_name)
            name = split_path[1] + name
            file_name = split_path[0]
            
        return name
            

    def get_preprocessing_combos(self):
        b = butterworth()
        s = ssfft()
        u = unembedded()
        process = [b.filter_signal, s.ssfft_fit]
        process_combinations = [(u.fit,)]
        
        process_combinations.extend(list(itertools.permutations(process)))

        for combs in process_combinations:
            for pros in combs:
                print(f'{pros.__qualname__}', end = "->")
            print('\n')

        return process_combinations
    
    def test_data(self, insts):
        
        results = []
        
        processing_combos = self.get_preprocessing_combos()
        
        classifiers = [SVM()]
        
        comparitors = [Mean_bag()]
        
        
        
        for process in processing_combos:
            #try:
                pre_inst = self.filter_pipe_line(insts, process)
            
                
                for comp in comparitors:
                    #try:
                        comp.compare_against_all(pre_inst)
                        for clf in classifiers:
                    
                            #try:
                                results.append((self.get_pipeline_name(process,comp,clf)
                                            ,clf.classify(self.get_ML_data(insts))))
                            #except Exception as e:
                                #print('Failed to process pipline')
                                #print(f'pipeline: -> {self.get_pipeline_name(process,comp,clf)}')
                                #print(e)
            
                   # except Exception as e:
                        #print('Failed to process pipline')
                        #print(f'pipeline: -> {self.get_pipeline_name(process,comp,none())}')
                        #print(e)
                    
            #except Exception as e:
                    #print('Failed to process pipline')
                    #print(f'pipeline: -> {self.get_pipeline_name(process,none(),none())}')
                    #print(e)
        return results
            
        
        
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
                        
    def filter_pipe_line(self, instructions, pipeline):

        for i in instructions:
            for p in pipeline:
                #print(f'Filtering {i.get_name()}')
                i.parse_callable(p)
            #print('debug') 

        return instructions
    
    def get_ML_data(self, instructions):
        label_list = []
        y_list = []
        X = []
        
        for i in range(len(instructions)):
            data = instructions[i].get_vecorized_data()
            y_list.append(np.zeros(len(data))+i)
            label_list.append(instructions[i].get_name())
            X.append(data)
            
        X = np.concatenate(X, axis = 0)
        y = np.concatenate(y_list, axis = 0)
        
        return X, y
    

    def get_preprocess_names(self, process_combination):
        
        name = ""
            
        for pros in process_combination:
                name = name + str(pros.__qualname__) + " > "
                
        return name
    
    def get_pipeline_name(self, process_combination, comparitor, classifier):
        
        name = ""
        
        name = name + self.get_preprocess_names(process_combination)
        name = name +' > '+ type(comparitor).__name__ +' > '+ type(classifier).__name__
        
        return name
    

    def get_file_list(self):
        direct = filedialog.askdirectory()
        
        return self.explore_folder(direct)
        
      
    def explore_folder(self, direct):
        
        files = []
        
        folders = []
        
        for root, dirs, fs in os.walk(direct):
            
            #for d in dirs:
            #    files.extend(self.explore_folder(os.path.join(root,d)))
                
            for f in fs:
                if 'test' in f and '.csv' in f:
                    print(os.path.join(root,f))
                    files.append(os.path.join(root,f))
                    
        return files
                    
        
class none():
    def __init__(self):
        self.__name__ = 'none'
        