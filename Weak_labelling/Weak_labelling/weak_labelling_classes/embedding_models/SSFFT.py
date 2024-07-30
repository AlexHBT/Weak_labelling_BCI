from mne import epochs
import scipy.fft
import numpy as np
import copy
class ssfft(object):
    
    epoch_length = None
    
    def __init__(self, epoch_length = 1):
        self.epoch_length = epoch_length
    
    def fit(self, data):
        pass
    
    def transform_bag(self, bag):
        
        transformed = []
        
        for s in bag:
            transformed.append(self.ssfft_fit(s))
            
        return transformed
        
    def ssfft_fit(self, signal):
        
        y = scipy.fft.fft(signal)
        
        
        P2 = np.absolute (y/y.shape[0])
        
        P1 = P2[1:int(y.shape[0]/2)+1]
        P1[2:-2] = 2*P1[2:-2]
        P1 = P1[int(8*self.epoch_length):int(15*self.epoch_length)] 
        return P1
        
        
        



