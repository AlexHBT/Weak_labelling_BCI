
import scipy as sci
import numpy as np
import copy
class ssfft(object):
    
    name = 'ssfft'
    
    epoch_length = None
    
    def ssfft(self,X):
        Y = sci.fft.fft(X, axis = 0)
        L = Y.shape[0]
        P2 = np.abs(Y/L)
        P1 = P2[:int(L/2)+1]
        P1[2:] = 2*P1[2:]
        x_freq = np.arange(int(L/2)+1)
        return np.square(P1[1:]), x_freq[1:]
        
        
        
    def fft_bag(self,bag):
        new_bag = []
        for i in bag:
            signal, axis = self.ssfft(i)
            new_bag.append(signal)
        #print(new_bag[0].shape)
        return new_bag


    def compress_bag(self,bag):
        new_values = []
        for i in bag:
            temp = []
            for j in range(i.shape[1]):
                temp.append(np.mean(i[:,j].reshape(2,-1, order = 'f'), axis = 0)[4:8])
            new_values.append(np.stack(temp, axis = 1))
        #print(new_values[0].shape)
        return new_values
    
    def log10(self,inst):
        return 10 * np.log10(inst)
    
    def get_bands(self, inst, args):
        sfft = self.ssfft(inst)[0]
        
        band_power = []

        if 'theta' in args:
            band_power.append(np.max(self.log10(sfft[3:7]), axis = 0))        

        if 'alpha' in args:
            band_power.append(np.max(self.log10(sfft[7:12]), axis = 0))
            
        if 'beta' in args:
            band_power.append(np.max(self.log10(sfft[12:30]), axis = 0))
          
        
            
        return np.stack(band_power, axis = 0)

      
    def get_bands_bag(self, bag, *args):
        
        new_data = []
        
        for i in bag:
            new_data.append(np.squeeze(self.get_bands(i,args)))
           
        return new_data
        
    
    def fft_compress(self,bag):
        return self.compress_bag(self.fft_bag(bag))
    
    def convert_data(bags):
        y = []
        total_data = []
        for b in range(len(bags)):
            y.append(np.zeros(len(bags[b])) + b)
            for i in bags[b]:
                total_data.append(i.reshape(1,-1))
        total_data = np.concatenate(total_data, axis = 0)
        return total_data, np.concatenate(y)
    
