class falten():
    
    def __init__(self):
        pass
    
    def flatten_list(self, value_list):
        for i in range(len(value_list)):
            value_list[i] = value_list[i].flatten()
            
        return value_list
            
    
    
        
        




