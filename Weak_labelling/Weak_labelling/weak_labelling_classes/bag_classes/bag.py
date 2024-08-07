import copy
import numpy as np
class Bag():
    
    _bag = None
    
    def __init__(self, data: list[np.ndarray] = None):
        
        self._bag = []
        
        if data == None:
            pass
        else:
            self.extend_bag(data)
            
    def add_example(self, example:list):
        self._bag.append(example)
        
    def extend_bag(self, bag:list[list]):
        self._bag.extend(bag)
        
    def get_example(self, index: int):
        return self._bag[index]
    
    def get_bag(self):
        return copy.copy(self._bag)
    
    def length(self):
        return(len(self._bag))
    
    def flatten_examples(self):
        for i in range(len(self._bag)):
            self._bag[i] = self._bag[i].flatten()
            
    def get_flattened_examples(self):
        
        new_bag = []
        
        for i in range(len(self._bag)):
            new_bag.append(self._bag[i].flatten())
            
        return new_bag 
            
    def apply_callable(self, function):
        for i in range(len(self._bag)):
            self._bag[i] = function(self._bag[i])
           
        #print('debug')
        
    def get_bag_mean(self):
        return np.mean(np.stack(self._bag, axis = 0), axis = 0)
        
        

    def remove_example(self, index: int):
        self._bag.pop(index)


    def remove_examples(self, indexes:[int]):
        
        indexes.sort()
        removed = 0
        
        for i in indexes:
            self._bag.pop(i-removed)
            removed+=1