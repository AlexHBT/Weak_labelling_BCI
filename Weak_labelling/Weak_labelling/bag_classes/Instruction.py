from .bag import Bag
import copy
class Instruction():
    
    _name = None
    
    _bags = []
    
    def __init__(self, bags: list[Bag] | Bag = None, name=''):
        
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
            
    def get_bags(self):
        return copy.copy(self._bags)
    
    def get_bag(self, index: int):
        return self._bags[index]
        
    def get_bags_length(self):
        return len(self._bags)
        




