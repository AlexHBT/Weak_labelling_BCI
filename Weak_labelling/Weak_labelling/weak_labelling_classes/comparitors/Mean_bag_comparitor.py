from ast import Compare
from importlib.metadata import packages_distributions
import numpy as np
from scipy.stats import f
class Mean_bag():
    
    threshhold = None
    
    def __init__(self, threshhold = 0.7):
        self.threshhold = threshhold
    
    def compare_against_all(self, instructions:list):
            
        for inst in instructions:
            inst.vectorize_data()
            
        values = []
        
        for i in range(len(instructions)):
            pos = instructions[i]
            negs = instructions[:]
            negs.pop(i)
            values.append(self.compare_inst(pos,negs))
           
        self.remove_instances(instructions, values)


    def compare_inst(self, pos_inst, neg_insts):
        
        neg_bags = []
        
        for n in neg_insts:
            neg_bags.extend(n.get_bags())
            
        pos_bag_values = []
        for p in range(pos_inst.get_bags_length()):
            pos_bag_values.append(self.compare_one_agaist_many(pos_inst.get_bag(p),neg_bags))
            
        return pos_bag_values
        

    def compare_one_agaist_many(self, pos_bag, neg_bags):
            
        pos_comparisons = []

        for n in neg_bags:
            pos_comparisons.append(self.compare_bag(pos_bag, n))
          
        means = np.mean(np.array(pos_comparisons), axis = 0)
        return means
    
              
    def compare_bag(self, pos_bag, neg_bag):
        difference = pos_bag.get_bag_mean() - neg_bag.get_bag_mean()
        
        comparisons = []
        
        for i in range(pos_bag.length()):
            comparisons.append(self.cosign_sim(pos_bag.get_example(i),
                                               difference 
                                               ))
            
        return comparisons
        
            
    def cosign_sim(self,vec1,vec2):
        return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
            
            
    def remove_instances(self, insts, values):
        
        for i in range(len(insts)):
            for b in range(insts[i].get_bags_length()):
                
                indexes = []
                
                for v in range(len(values[i][b])):
                    if values[i][b][v] < self.threshhold:
                        indexes.append(v)
                                
                print(f'removed {len(indexes)} values, from bag {b} for {insts[i].get_name()}')
                insts[i].get_bag(b).remove_examples(indexes)
                
        
        
        
        
        
            
            