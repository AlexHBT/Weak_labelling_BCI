import copy
class Bag():
    
    _bag = []
    
    def __init__(self, data: list[list] = None):
        
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
            
    def apply_callable(self, function):
        for i in range(len(self._bag)):
            self._bag[i] = function(self._bag[i])
        
        
        




