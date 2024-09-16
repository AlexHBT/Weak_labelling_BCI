import scipy as sci
import numpy as np

class Bag_filters():
    
    def __init__(self):
        pass

    


    def small_lapalcian(self,epoch):
        layout = [[-1,-1,0,-1,-1],
                  [1,2,3,4,5],
                  [6,7,8,9,10],
                  [11,12,13,14,15]]

        epoch_copy = epoch[:]

        for i in range(len(layout)):
            for j in range(len(layout[i])):
                temp = []
                #print(layout[i][j])
                if layout[i][j] == -1:
                    pass
                else:
                    if i+1 != len(layout):
                        if layout[i+1][j] != -1:
                            temp.append(epoch[:,layout[i+1][j]])
                    
                    if i-1 > -1:
                        if layout[i-1][j] != -1:
                            temp.append(epoch[:,layout[i-1][j]])

                    if j-1 > -1:
                        if layout[i][j-1] != -1:
                            temp.append(epoch[:,layout[i][j-1]])

                    if j+1 != len(layout[0]):
                        if layout[i][j+1] != -1:
                            temp.append(epoch[:,layout[i][j+1]])
                        
                    m = np.mean(np.stack(temp, axis = 0), axis = 0)/4
                    epoch_copy[:,layout[i][j]] = epoch[:,layout[i][j]] - m
    
        return epoch_copy

    def sl_bag(self,bag):
        new_bag = []
        for b in bag:
            new_bag.append(self.small_lapalcian(b))
        return new_bag
    
    def butterworth(self, epoch):
        b,a = sci.signal.butter(4, [6,20], btype = 'bandpass', fs = 512)
        return sci.signal.filtfilt(b,a,epoch, axis = 0)
    
    def butter_bag(self,bag):
        new_bag = []
        for x in bag:
            new_bag.append(self.butterworth(x))
        return new_bag
    
    def filter_bag(self, bag):
        new_bag = []
        for x in bag:
            new_bag.append(self.butterworth(self.small_lapalcian(x)))
        return new_bag
    