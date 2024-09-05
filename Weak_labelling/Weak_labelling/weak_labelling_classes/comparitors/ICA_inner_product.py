import numpy as np
import scipy as sci
import copy 
from sklearn.decomposition import PCA
from ..bag_classes.bag import Bag

class ICA_inner():
    
    ints_dict = {
        '1': 'left',
        '2': 'right',
        '3': 'forward'
    }
    instruction_type = None
    current_negative = None
    
    threshold = 0.7
    
    def compare_against_all(self, instructions:list):
            
        values = []
        
        for i in range(len(instructions)):
            self.instruction_type = i
            pos = instructions[i]
            negs = instructions[:]
            negs.pop(i)
            values.append(self.compare_inst(pos,negs)) 
            
        self.remove_instances(instructions, values)
        

        
    def compare_inst(self, positive, negative):
        values = []
        for n in negative:
            values.append([])
            self.current_negative = self.get_instruction_int(n)
            for i in range(positive.get_bags_length()):
                values[-1].append(self.compare_bags(
                                    positive.get_bag(i),self.combine_bags(n.get_bags())))
                
        return self.mean_values(values)
                
            
    def combine_bags(self, bags):
        new_bag = Bag()
        
        for b in bags:
            new_bag.extend_bag(b.get_bag())
            
        return new_bag
        
    def mean_values(self, values):
        if len(values) == 1:
            return values[0]
         
        new_values = []
        
        for i in range(len(values[0])):
            new_values.append([])
            for j in range(len(values[0][0])):
                new_values[-1].append(np.mean([values[0][i][j], values[1][i][j]]))
            
        return new_values
    
    def compare_bags(self, positive, negative):
        
        pos = self.process_bag(positive.get_bag(), int(self.instruction_type+1))
        neg = self.process_bag(negative.get_bag(), self.current_negative)

        X, y = self.PCA_data([pos,neg])
        
        pca_bags = self.split_bags(X,y)
        return self.compare_examples(pca_bags)
        
    def process_bag(self, bag, inst):
        f_bag = self.filter_bag(bag) # might have to change to bag
        c_bag = self.convert_bag(f_bag, inst)
        return self.compress_bag(self.fft_bag(c_bag))
        

    def filter_bag(self,bag):
        b,a = sci.signal.butter(4, [6,20], btype = 'bandpass', fs = 512)

        new_bag = []
        for x in bag:
            new_bag.append(sci.signal.filtfilt(b,a,x, axis = 0))

        return new_bag

    def whitten(self,epoch):
        E, D = np.linalg.eig(np.cov(epoch.T))
        white_epoch =  np.matmul(sci.linalg.sqrtm(np.linalg.pinv(D)) * E.T,  epoch.T)
        return white_epoch.T, E, D
    
    def get_mixing(self,X, E, D):
        #print(D.shape)
        V, s, u = np.linalg.svd(np.matmul(np.multiply(
                        np.sum(np.multiply(X,X), axis = 0), X).T, X))
        #print (V.shape)
        W = V * sci.linalg.sqrtm(np.linalg.pinv(D)) * E.T
        return W
    
    def ICA_data(self,epoch):
        white, E, D = self.whitten(epoch)
        #print (white.shape)
        mixing = self.get_mixing(white, E, D)
        sources = np.matmul(epoch, mixing)
        return sources, mixing
    
    def ssfft(self,X):
        Y = sci.fft.fft(X, axis = 0)
        L = Y.shape[0]
        P2 = np.abs(Y/L)
        P1 = P2[:int(L/2)+1]
        P1[2:] = 2*P1[2:]
        x_freq = np.arange(int(L/2)+1)
        return P1[1:], x_freq[1:]
    
    def fft_bag(self,bag):
        new_bag = []
        for i in bag:
            signal, axis = self.ssfft(i)
            new_bag.append(signal)
        #print(new_bag[0].shape)
        return new_bag   
    
    def get_comparison(self,inst):
        layout = []
        if inst == 1:
            layout = [[0,0,0,0,0],
                      [0,0,0,1,0],
                      [0,0,0,0,0]]

        elif inst == 3:
            layout = [[0,0,0,0,0],
                      [0,0,1,0,0],
                      [0,0,0,0,0]]

        elif inst == 2:
            layout = [[0,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,0,0,0]]

        return np.concatenate(([0], np.array(layout).flatten())) 
    
    def compare_values(self,perfect,example):
        perfect = perfect* np.max(example)
        #v2 = v2/np.linalg.norm(v2)
        #print(v1)
        #print(v2)
        return np.inner(perfect,example)
    
    def get_components(self,example, instruction, n_comps = 3):
        sources, mix = self.ICA_data(example)
        Im = np.linalg.inv(mix).real

        values = []
        perfect_example = self.get_comparison(instruction)

        for i in range(Im.shape[0]):
            values.append(abs(self.compare_values(perfect_example, Im[:,i])))

    
    
        #prints out values
        #order_list = copy.deepcopy(values)
        #order_list.sort()
        #print(order_list[:3])
    
        max_indexes = []
        for i in range(n_comps):
            max_indexes.append(np.argmax(values))
            values.pop(max_indexes[-1])

        #plots the main value
        #plot_comps(Im[:,max_indexes],instruction)
    
        comps = example[:, max_indexes]
        
        return comps
    
    def convert_bag(self,bag:[np.ndarray], inst):
        new_comps = []
        for i in bag:
            new_comps.append(self.get_components(i, inst))

        return new_comps
    
    def compress_bag(self,bag):
        new_values = []
        for i in bag:
            temp = []
            for j in range(i.shape[1]):
                temp.append(np.mean(i[:,j].reshape(2,-1, order = 'f'), axis = 0)[4:10])
            new_values.append(np.stack(temp, axis = 1))
        #print(new_values[0].shape)
        return new_values
    
    def PCA_data(self, bags:[list]):
        y = []
        total_data = []
        for b in range(len(bags)):
            y.append(np.zeros(len(bags[b])) + b)
            for i in bags[b]:
                total_data.append(i.reshape(1,-1))
        total_data = np.concatenate(total_data, axis = 0)
        pca = PCA(2)
        total_data = pca.fit_transform(total_data)
        return total_data, np.concatenate(y)
    
    def split_bags(self , X, y):
        unq = np.unique(y)
        bags = []
        for u in unq:
            bags.append([])
            
        for i in range(y.shape[0]):
            for val in range(len(unq)):
                if y[i] == unq[val]:
                    bags[val].append(X[i,:])
    
        return bags
    
    def cos_sim(self,a,b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
    def get_bag_means(self, bags):
        means = []
        for b in bags:
            means.append(np.mean(np.stack(b, axis = 0), axis = 0))
            
        return means
    
    def get_seperable_vec(self, bags):
        means = self.get_bag_means(bags)
        return means[0] - means[1]
    
    def compare_examples(self, bags):
        sep = self.get_seperable_vec(bags)
        values = []
        pos = bags[0]
        for i in range(len(pos)):
           values.append(self.cos_sim(sep, pos[i]))
        return values
    
        
    
    def convert_data(bags):
        y = []
        total_data = []
        for b in range(len(bags)):
            y.append(np.zeros(len(bags[b])) + b)
            for i in bags[b]:
                total_data.append(i.reshape(1,-1))
        total_data = np.concatenate(total_data, axis = 0)
        return total_data, np.concatenate(y)
    
    def remove_values(self, bag: Bag, values):
        examples = []
        
        for i in values:
            examples.append(bag.get_example(i))
            
        bag.set_bag(examples)
        

    def remove_instances(self, insts, values):
        
        for i in range(len(insts)):
            for b in range(insts[i].get_bags_length()):
                
                indexes = []
                
                for v in range(len(values[i][b])):
                    if values[i][b][v] < self.threshold:
                        indexes.append(v)
                                
                print(f'removed {len(indexes)} values, from bag {b} for {insts[i].get_name()}')
                insts[i].get_bag(b).remove_examples(indexes)
                

    def get_instruction_int(self, inst):
        
        match inst.get_name().lower():
            case 'left':
                return 1
            case 'right':
                return 2
            case 'forward':
                return 3
        
            
         