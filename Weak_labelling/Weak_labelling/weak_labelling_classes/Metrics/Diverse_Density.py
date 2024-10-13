import numpy as np
class diverse_denisty():
    def __init__(self):
        pass
    
    

    def test_position(self,pos_bags, neg_bags, x):
        pos_bag_probs = []
        for bag in pos_bags:
            pos_bag_probs.append(self.bag_prob(bag,x))

        neg_bag_probs = []
        for bag in neg_bags:
            neg_bag_probs.append(self.bag_prob(bag,x))

        return np.prod(pos_bag_probs)*np.prod(neg_bag_probs)
        

    def bag_prob(self,bag,x):
        probs = []
        for inst in bag:
            probs.append(1-self.prob(inst,x))

        return 1 - np.prod(probs)

    def prob(self,instance,x):
        return np.exp(-(np.abs(instance-x)**2))
