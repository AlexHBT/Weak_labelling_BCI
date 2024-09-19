import sklearn as sk
import numpy as np

class outlier_dection():
    def __init__(self):
        pass
    
    def elliptic(self, bag: list | np.ndarray):

        if type(bag).__name__ == 'list':
            temp_bag = np.stack(bag, axis = 0)
            temp_bag = self._flattend_array(temp_bag)
            est = sk.covariance.EllipticEnvelope()
            vals = est.fit_predict(temp_bag)
            new_bag = []
            for i in range(len(vals)):
                if vals[i] == 1:
                    new_bag.append(bag[i])
                    
            print(f'anomally_removal: ({len(new_bag)}/{len(bag)})')
            return new_bag
                    

        else:
            temp_bag = bag[:]
            temp_bag = self._flattend_array(temp_bag)
            est = sk.covariance.EllipticEnvelope()
            vals = est.fit_predict(temp_bag)
            indx = []
            for i in range(len(vals)):
                if vals[i] == 1:
                    indx.append(i)
                
            print(f'anomally_removal: ({len(indx)}/{bag.shape[0]})')

            return bag[indx,...]

            

    def _flatten_bag(self,bag):
        new_bag = []
        for b in bag:
            new_bag.append(b.flatten())
        return new_bag 
        
    def _flattend_array(self, arr):
        return arr.reshape((arr.shape[0],-1))
        
    def std_remover(self,bag):
        
        if type(bag).__name__ == 'list':
            temp_bag = np.stack(bag, axis = 0)
            temp_bag = self._flattend_array(temp_bag)
            vals = self._std_evaluator(temp_bag,3)
            new_bag = []
            for i in vals:
                
                new_bag.append(bag[i])
                    
            #print(f'anomally_removal: ({len(new_bag)}/{len(bag)})')
            return new_bag
                    

        else:
            temp_bag = bag[:]
            temp_bag = self._flattend_array(temp_bag)

            indx = self._std_evaluator(temp_bag,3)
                
            #print(f'anomally_removal: ({len(indx)}/{bag.shape[0]})')

            return bag[indx,...]
        
    def _std_evaluator(self, bag, dist):
        
        r_indx = []

        for i in range(bag.shape[1]):
            m = np.mean(bag[:,i])
            std = np.std(bag[:,i])
            r = std * dist
            
            for j in range(bag.shape[0]):
                if bag[j,i] > m + r or bag[j,i] < m - r:
                    if j not in r_indx:
                        r_indx.append(j)
                        
        vals = []
                        
        for i in range(bag.shape[0]):
            if i not in r_indx:
                vals.append(i)
                
        return vals
            
            
            
             
            
            
        