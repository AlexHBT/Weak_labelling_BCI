from typing import Self
import numpy as np
import os
from .Participant_test import participant_test
import pandas as pd

class Simulated_Training():
    
    def __init__(self):
        pass
    
    def start(self, direc):
        participants, names  = self.get_participants(direc)
        
        output_dir = 'D:/Weak_labelling_results/'
        
        participant_accuracies = []
        
        for i in range(len(participants)):
            
            
                participant_accuracies.append(
                    self.test_participant(participants[i]))
            
                df = pd.DataFrame(participant_accuracies[-1])
                df.to_csv(
                    os.path.join(os.path.abspath(output_dir), f'{names[i]}_accuracies.csv'))
            
            
                
        
            
            
    
    def get_participants(self, direct):
        participants = []
        participants_names = []
        for i in next(os.walk(direct))[1]:
            participants_names.append(i)
            participants.append(os.path.join(
                os.path.abspath(direct),i))
            
        return participants, participants_names
        
    
    def test_participant(self, participant_dir):
        
        pt = participant_test()
        
        files = self.get_files(participant_dir)
        
        train_files = []
        test_files = []
        
        accuracies = []
        
        for i in range(len(files)-1):
            try:
                train_files = []
                test_files = []
                for j in range(3):
                    if i-j<0:
                        break
                    else:
                        train_files.append(files[i-j])
                test_files.append(files[i+1])
            
                accuracies.append(pt.start(train_files, test_files))
                
            except:
                print('Failed to get results')
        return accuracies
        
    def run_test(self, train_files, test_files):
        pass
        
     
    
    def get_files(self, participant_dir):
        
        files = []
        
        data = next(os.walk(participant_dir))
        
        
            
        for f in data[2]:
            if 'test' in f and '.csv' in f:
                files.append(os.path.join(os.path.abspath(data[0]),f))
                    
        for d in data[1]:
            files.extend(self.get_files(
                os.path.join(os.path.abspath(data[0]),d)))
                
        return files
                
            
        


