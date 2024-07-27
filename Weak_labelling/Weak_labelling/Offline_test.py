import pandas as pd
import numpy as np
from data_loaders.CSV_loader import CSV_loader

from tkinter import filedialog


 
class offline_test():
    
    def __init__(self):
        pass
    
    def start(self):
        
        files = list(filedialog.askopenfilenames())
        print(files)
        
        
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
    
    
    
            
