import numpy as np
class ica_metrics(object):
    
    def __init__(self):
        pass
    
    def data_loss(self, bag_size:[int], reduced_size:[int]):
        N = len(bag_size)
        b = np.array(bag_size)
        r = np.array(reduced_size)
        
        #return np.sum(np.abs(np.log(np.divide((b-r),b)))/N)
        return np.sum(np.abs(np.log(np.divide(r,b)))/N)



