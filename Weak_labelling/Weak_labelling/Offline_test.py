import pandas as pd
import numpy as np
from data_loaders.CSV_loader import CSV_loader

from tkinter import filedialog
from Filters.Butterworth import butterworth
from embedding_models.SSFFT import ssfft
from comparitors.Mean_bag_comparitor import Mean_bag

 
class offline_test():
    
    def __init__(self):
        pass
    
    def start(self):
        
        files = list(filedialog.askopenfilenames())
        bags = self.load_file_data(files)
        filtered_bags = self.filter_signals(bags)
        mbc = Mean_bag()
        selected = mbc.compare_against_all(filtered_bags[1:])
        print ('stop')
        
    def load_file_data(self, files):
        csvl = CSV_loader()
        
        bags = []
        
        for f in files:
            bags.append(csvl.load_bag_CSV(f))
            
        instructions = [[],[],[],[]]
            
        for b in bags:
            for i in range(len(instructions)):
                instructions[i].extend(b[i])
                
        return instructions
    
    def filter_signals(self,bags):
        
        filtered_signals = []
        
        butter = butterworth()
        fft = ssfft()
        
        for c in bags:
            filtered_signals.append([])
            for b in c:
                filtered_signals[-1].append(fft.transform_bag(butter.filter_signal(b)))
                b.clear()
            c.clear()
                
        return filtered_signals
            
    
    

    
    
    
            
