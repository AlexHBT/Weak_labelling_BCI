import pandas as pd
import numpy as np
from ..bag_classes import (bag, Instruction)

class CSV_loader():
    
    freq = None
    overlap = None
    epoch_length = None
    
    def __init__(self, freq:int = 512, overlap:int = 0.5, epoch_length:int = 1):
        
        self.freq = freq
        self.overlap = overlap
        self.epoch_length = epoch_length
    
    def load_standard_CSV(self,data_dir):
        
    

        df = pd.read_csv(data_dir)

        data_columns = []
        event_columns = []
    
        for name in df.columns:
            if "Channel" in name:
                data_columns.append(name)
            if "Event" in name:
                event_columns.append(name) 

        data_df = df[data_columns]
        event_df = df[event_columns]

   
    
    
        events = event_df["Event Id"]

        triggers = events.index[events.notna()].tolist()

        trig_values = events.iloc[triggers].tolist()

        data = data_df.to_numpy()
        #print(triggers)
        #print(trig_values)

        instructions = [[],[],[],[]]

        for i in range(len(triggers)-1):

            instructions[int(trig_values[i])].extend(self.cut_instance(data[triggers[i]:triggers[i+1],...]))

        return instructions
    

    def load_bag_CSV(self,data_dir):
        
        df = pd.read_csv(data_dir)

        data_columns = []
        event_columns = []
    
        for name in df.columns:
            if "Channel" in name:
                data_columns.append(name)
            if "Event" in name:
                event_columns.append(name) 

        data_df = df[data_columns]
        event_df = df[event_columns]

   
    
    
        events = event_df["Event Id"]

        triggers = events.index[events.notna()].tolist()

        trig_values = events.iloc[triggers].tolist()

        data = data_df.to_numpy()
        #print(triggers)
        #print(trig_values)

        instructions = [[],[],[],[]]

        for i in range(len(triggers)-1):

            instructions[int(trig_values[i])].append(self.cut_instance(data[triggers[i]:triggers[i+1],...]))

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
    
 
    
        




