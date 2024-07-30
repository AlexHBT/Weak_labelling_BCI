from ast import Compare
from importlib.metadata import packages_distributions
import numpy as np
from scipy.stats import f
class Mean_bag():
    
    def __init__(self):
        pass
    
    def compare_against_all(self, bags:list):
            
        complete_bags = []
        
        for i in range(len(bags)):
            bag_copy = bags[:]
            pos_bags = bag_copy.pop(i)
            
            complete_bags.append([])
            
            for p in pos_bags:
               complete_bags[-1].append(self.compare_bags(p, bag_copy)) 
            
        return complete_bags
        

    def compare_against_other(self, bags:list):
        pass


    def compare_bags(self, pos_bag, neg_bags):
        
        negative_bags = []
        
        for n in neg_bags:
            negative_bags.extend(n)
        neg_bags.clear()
        
        values = []
        
        for i in pos_bag:
            temp_values = []
            for j in negative_bags:
                temp_values.append(self.compare_bag_dot_product(pos_bag,j))
            values.append(np.array(temp_values))   

        values = np.stack(values, axis = 0)
        
        mean_values = np.mean(np.mean(values, axis = 1),axis = 1)
                
        mean_values = np.mean(mean_values, axis = 1)

        final_bag = []
        
        for i in range(len(pos_bag)):
            if mean_values[i] > 0.7:
                final_bag.append(pos_bag[i])
                
        return final_bag
             
            
            
        
    def compare_bag_dot_product(self, pos_bag, neg_bag):
        
        
        
        mean_vector = np.mean(pos_bag) - np.mean(neg_bag)
        
        vals = []
        
        for i in range(len(pos_bag)):
            vals.append(np.dot(pos_bag[i], mean_vector))
        
        return vals
    
    
               
              
        
            
            
            
            
    
        
        
        
        
        
            
            