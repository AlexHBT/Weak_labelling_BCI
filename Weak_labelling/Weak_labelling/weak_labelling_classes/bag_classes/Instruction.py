from .bag import Bag
import copy
import numpy as np
class Instruction():
    
    _name = None
    
    _bags = None
    
    def __init__(self, bags: list[Bag] | Bag = None, name=''):
        
        self._bags = []
        
        self._name = name
        
        if bags == None:
            pass
        else:
            self.add_bag(bags)
        
    def add_bags(self, bags: list[Bag] | Bag):
        
        if type(bags).__name__ == 'list':
            self._bags.extend(bags)
        else:
            self._bags.append(bags)
            
    def create_bag(self, data:[np.ndarray]):
        
        self._bags.append(Bag(data))
            
    def get_bags(self):
        return copy.copy(self._bags)
    
    def get_bag(self, index: int):
        return self._bags[index]
        
    def get_bags_length(self):
        return len(self._bags)
    
    def parse_callable(self, function):
               
        #print(f'parsing{self._name}') 
        for b in self._bags:
            b.apply_callable(function) 
        #print('instruction parse_callable')
            
    def clear_bags(self):
        self._bags.clear()
        
    def get_name(self):
        return self._name
        
            
            
        
        




