import numpy as np
from ..bag_classes.bag import Bag
from ..Filters.Bag_filters import Bag_filters
from ..embedding_models.SSFFT import ssfft
from ..classifiers import SVM

class bci_tvr(object):
    dtype = None
    
    def __init__(self, dtype: str = 'bag'):
        self.dtype = dtype
        
    def process_and_classify(self, data):
        data = self.check_dtype(data)
            
        X, y = self.convert_to_ml_data(
                self.process_data(data))
        
        return self.classify(X,y)
        
            
    def check_dtype(self, data):
        if self.dtype == 'inst':
            new_data = []
            for i in data:
                new_data.append(self.combine_bags(i.get_bags()).get_bag())
            
        else:
            return data
    
    def combine_bags(self,bags):
        new_bag = Bag()
        for b in bags:
            new_bag.extend_bag(b.get_bag())
        return new_bag

    def process_data(self, data):
        bf = Bag_filters()
        s = ssfft()
        for i in range(len(data)):
            data[i] = s.fft_compress(bf.filter_bag(data[i]))
            
        return data

    def convert_to_ml_data(self,bags):
        y = []
        total_data = []
        for b in range(len(bags)):
            y.append(np.zeros(len(bags[b])) + b)
            for i in bags[b]:
                total_data.append(i.reshape(1,-1))
        total_data = np.concatenate(total_data, axis = 0)
        return total_data, np.concatenate(y)

    def stack_data(self,bag):
        return np.stack(bag, axis = 0)
    
    def flatten_data(self,bag):
        new_values = []

        for i in bag:
            new_values.append(i.flatten())
        return new_values
    
    def stack_flatten(self, bag):
        return self.stack_data(self.flatten_data(bag))
    
    def classify(self, X,y):
        cl = SVM.SVM()
        return cl.classify_fold_accuracy(X,y)
        