import numpy as np
import mne
from ..bag_classes import (bag, Instruction)
import copy
import pandas as pd

class bci_comp_loader():
    
    freq = None
    overlap = None
    epoch_length = None
    
    def __init__(self):
        self.freq = 250
        self.overlap = 0.5
        self.epoch_length = 1
        mne.set_log_level('CRITICAL')
        
    def load_data_bag(self, path):
        
        eeg_channels = ['EEG-Fz', 'EEG-0', 'EEG-1','EEG-2','EEG-3','EEG-4', 'EEG-C3','EEG-6',	'EEG-Cz',	'EEG-7',	'EEG-C4', 'EEG-9',	'EEG-10',	'EEG-11',	'EEG-12',	'EEG-13']
        

        data = mne.io.read_raw_gdf('D:/BCI comp data set 2a/A01T.gdf')
        data_df = data.to_data_frame()[eeg_channels]
        
       
        events, event_id = mne.events_from_annotations(data)
        
        instructions = [Instruction.Instruction(name = 'Left'),
                        Instruction.Instruction(name = 'Right'),
                        Instruction.Instruction(name = 'Forward')]
        
        #print(f'Left instructions {np.count_nonzero(events[:,2] == 7)}')
        #print(f'Right instructions {np.count_nonzero(events[:,2] == 8)}')
        #print(f'Forward instructions {np.count_nonzero(events[:,2] == 9)}')
    
        #print(events[:,2])

        for i in range(events.shape[0]-1):

            if events[i,2] == 7:
                instructions[0].create_bag(self.cut_instance(data_df.to_numpy()[events[i,0]:events[i+1,0],:]))
            
            
            if events[i,2] == 8:
                instructions[1].create_bag(self.cut_instance(data_df.to_numpy()[events[i,0]:events[i+1,0],:]))
            
            if events[i,2] == 9:
                instructions[1].create_bag(self.cut_instance(data_df.to_numpy()[events[i,0]:events[i+1,0],:]))
            
           

        return instructions
    
       
        

    def cut_instance(self,data):

        temp_list = []

        pointer = 0
        jump = int(self.freq * self.overlap)
        epoch_len = int(self.freq * self.epoch_length)

        while(pointer+epoch_len < data.shape[0]):
            temp_list.append(data[pointer:pointer+epoch_len,:])
            pointer += jump
    

        return(temp_list)
    


    
