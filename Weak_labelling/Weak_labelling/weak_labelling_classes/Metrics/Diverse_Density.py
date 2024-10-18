import numpy as np
class diverse_denisty():
    def __init__(self):
        pass
    
    

    def test_position(self,pos_bags, neg_bags, x):
        pos_bag_probs = []
        for bag in pos_bags:
            pos_bag_probs.extend(self.pos_bag_prob(bag,x))

        neg_bag_probs = []
        for bag in neg_bags:
            neg_bag_probs.extend(self.neg_bag_prob(bag,x))

        return np.sum(pos_bag_probs)+np.sum(neg_bag_probs)
        

    def pos_bag_prob(self,bag,x):
        probs = []
        for inst in bag:
            probs.append(self.pos_prob(inst,x))

        return probs
    
    def neg_bag_prob(self,bag,x):
        probs = []
        for inst in bag:
            probs.append(self.neg_prob(inst,x))

        return probs


    def pos_prob(self,instance,x):
        return np.sum(-np.exp(np.abs(instance-x))**2)

    def neg_prob(self,instance,x):
        return np.sum(-1/(np.exp(np.abs(instance-x))**2))
        