import pandas as pd
import numpy as np
from weak_labelling_classes.data_loaders.CSV_loader import CSV_loader

from tkinter import filedialog
from weak_labelling_classes.Filters.Butterworth import butterworth
from weak_labelling_classes.embedding_models.SSFFT import ssfft
from weak_labelling_classes.comparitors.Mean_bag_comparitor import Mean_bag

 
class offline_test():
    
    instructions = None
    
    def __init__(self):
        pass
    
    def start(self):
        
        files = list(filedialog.askopenfilenames())
        self.instructions  = self.load_file_data(files)
        print('Loaded instructions')
        self.filter_pipe_line()
        self.print_lengths()
        
        
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
        pipeline = [b.filter_signal, s.ssfft_fit]
        
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
            
        