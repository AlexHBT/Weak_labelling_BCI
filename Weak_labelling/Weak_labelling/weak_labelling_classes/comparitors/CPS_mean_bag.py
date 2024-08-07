from mne.decoding import CSP
import numpy as np

class CSP_mean_bag():
    
    comps = None

    def __init__(self, components):
        self.comps = components
        
    
    def compare_against_all(self, instructions:list):
            
        values = []
        
        for i in range(len(instructions)):
            pos = instructions[i]
            negs = instructions[:]
            negs.pop(i)
            values.append(self.compare_inst(pos,negs)) 
      
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
        
        pos, neg = self.embed_CSP(pos_bag.get_bag(), neg_bag.get_bag())
        
        pos = self.flatten_array(pos)
        neg = self.flatten_array(neg)
        
        difference = np.mean(np.stack(pos, axis = 0), axis = 0) - np.mean(np.stack(neg, axis = 0), axis = 0)
        
        
        comparisons = []
        
        for i in range(len(pos)):
            comparisons.append(self.cosign_sim(pos[i],
                                               difference 
                                               ))
            
        return comparisons
        
            
    def cosign_sim(self,vec1,vec2):
        return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    
    def inner_prod(self, vec1, vec2):
        return np.dot(vec1, vec2)
    
    def embed_CSP(self, pos_bag, neg_bag):
        
        c = CSP(n_components = self.comps)
        pos = np.swapaxes(np.stack(pos_bag, axis = 0),1,2)
        neg = np.swapaxes(np.stack(neg_bag, axis = 0),1,2)
        
        pos_y = np.zeros(pos.shape[0])
        neg_y = np.ones(neg.shape[0])
        
        X = np.concatenate((pos,neg), axis = 0)
        y = np.concatenate((pos_y,neg_y), axis = 0)
        
        c =  CSP(n_components = self.comps).fit(X,y)
        
        
        pos = []
        neg = []
        
        for i in pos_bag:
            pos.append(c.transform(i.T))
        for i in neg_bag:
            neg.append(c.transform(i.T))
           
        return pos, neg
        

            
    def remove_instances(self, insts, values):
        
        for i in range(len(insts)):
            for b in range(insts[i].get_bags_length()):
                
                indexes = []
                
                for v in range(len(values[i][b])):
                    if values[i][b][v] < self.threshhold:
                        indexes.append(v)
                                
                print(f'removed {len(indexes)} values, from bag {b} for {insts[i].get_name()}')
                insts[i].get_bag(b).remove_examples(indexes)
                
    def flatten_array(self,arr):
        
        temp = []
        for a in arr:
            temp.append(a.flatten())
            
        return temp 