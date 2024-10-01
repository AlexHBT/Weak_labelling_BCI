from typing import Self
import numpy as np
import os
from .Participant_test import participant_test
import pandas as pd
from ..comparitors.ICA_inner import ICA_inner_2
import math


class Simulated_Training():
    
    train_history = 1 # number of previous files included in data 
    
    def __init__(self):
        pass
    
    def start(self, direc):
        participants, names  = self.get_participants(direc)
        
        #output_dir = 'D:/Weak_labelling_results/'
        output_dir = 'E:/Alex/Weak labelling test/Results/'
        
        participant_accuracies = []
        
        for i in range(len(participants)):
            
                print (f'testing participant {i+1}: {names[i]}') 
            
                participant_accuracies.append(
                    self.test_participant(participants[i],names[i]))
            
                df = pd.DataFrame(participant_accuracies[-1], columns = ICA_inner_2('','').get_data_columns())
                
                df.to_csv(
                    os.path.join(os.path.abspath(output_dir), f'{names[i]}_accuracies.csv'))
            
            
                
        
            
            
    
    def get_participants(self, direct):
        
        participants = self.get_files(direct)
                    
        participants_names = []

        for i in participants:
            participants_names.append(os.path.splitext( os.path.basename(i))[0])
            
        return participants, participants_names
        
    
    def test_participant(self, participant_file, part_name):
        self.print_progress(0,1)
        
        pt = participant_test()
        
        accuracies = []
                
        #try:
            
            
        accuracies.append(pt.start_2([participant_file],part_name, 1))
                
        #except:
            #print('Failed to get results')
        
        self.print_progress(1,1)
        return accuracies
        
    def run_test(self, train_files, test_files):
        pass
        
     
    
    def get_files(self, participant_dir):
        
        files = []
        
        data = next(os.walk(participant_dir))
        
        
            
        for f in data[2]:
            if 'T' in f and '.gdf' in f:
                files.append(os.path.join(os.path.abspath(data[0]),f))
                    
        #for d in data[1]:
            #files.extend(self.get_files(
                #os.path.join(os.path.abspath(data[0]),d)))
                
        return files
                
            
    
    def print_progress(self,i, length, bar_length = 20):
        if i == 0:
            bar = '-' * bar_length
            print(f'session ({i}/{length}) [' + '-'*bar_length+ ']', end = '\r')
            
        elif i == length:
            print(f'session ({i}/{length}) [' + '='*bar_length+ '] done', end = '\n')
        else:
            percent = i/length
            eq = '=' * int(bar_length*percent)
            bar = '-' * int(bar_length-int(bar_length*percent))
            print(f'session ({i}/{length}) [{eq}{bar}]', end = '\r')
            
    


