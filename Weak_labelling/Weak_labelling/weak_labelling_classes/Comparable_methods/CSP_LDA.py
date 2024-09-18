from mne.decoding import CSP
import mne
import numpy as np
from ..bag_classes.bag import Bag
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from ..Filters.Bag_filters import Bag_filters
from ..embedding_models.SSFFT import ssfft

class csp_classifier(object):
    
    def __init__(self, dtype: str = 'bag'):
        self.dtype = dtype
        
        mne.set_log_level('CRITICAL')

    def process_and_classify(self, data):
        data = self.check_dtype(data)
            
        X, y = self.convert_to_ml_data(
                self.process_data(data))
        
        return self.classify(X,y)
    
    
    def convert_to_ml_data(self,bags):
        y = []
        total_data = []
        for b in range(len(bags)):
            y.append(np.zeros(len(bags[b])) + b)
            for i in bags[b]:
                total_data.append(i)
        total_data = np.stack(total_data, axis = 0)
        return np.swapaxes(total_data,1,2), np.concatenate(y)
        
            
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
    
    def classify(self, X,y):
        clf = Pipeline([("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)), 
                        ("LDA", LinearDiscriminantAnalysis())])

        return np.mean(cross_val_score(clf, X, y,error_score='raise'))

