import numpy as np
class sim_measures():
    
    def __init__(self):
        pass
       
    def signed_inner(self,example,arr):
        return -1 * (np.abs(np.dot(example,arr)) * np.sum(np.multiply(example,arr)))
    
    def inner(self,v1,v2):
        return np.inner(v1,v2)
    
    def cos_sim(self,a,b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
    def distance(self, v1,v2):
        return -1 * np.abs(np.sum((v1-v2)))
    
    def scalled_inner(self, v1,v2):
        v1 = v1* np.max(v2)
        v2 = v2/np.linalg.norm(v2)
        #print(v1)
        #print(v2)
        return np.inner(v1,v2)

    