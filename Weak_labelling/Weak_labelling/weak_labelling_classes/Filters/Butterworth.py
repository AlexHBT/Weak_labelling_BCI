import scipy.signal
class butterworth(object):
    
    filt_b = None
    filt_a = None 
    
    def __init__(self, upper = 17, lower = 6, order = 4):
        self.filt_b, self.filt_a  =  scipy.signal.butter(order, [lower,upper],btype='bandpass', fs = 512)
        
    def filter_signal(self, signal):
        
        if type(signal).__name__ == 'list': 
            
            filtered_signals = []
            
            for i in signal:
               filtered_signals.append(
                   scipy.signal.filtfilt(self.filt_b, self.filt_a, i, axis = 0))
            
            return filtered_signals
            
        else:
            if len(signal.shape) == 3:
                return scipy.signal.filtfilt(self.filt_b, self.filt_a, signal, axis = 1)
            
            else:
                return scipy.signal.filtfilt(self.filt_b, self.filt_a, signal, axis = 0)
            
        



