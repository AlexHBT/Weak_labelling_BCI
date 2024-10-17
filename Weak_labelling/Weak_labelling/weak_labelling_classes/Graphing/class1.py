import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import umap
import matplotlib.patches

class bag_scatter_graphs(object):
    
    graph_dir = 'D:/Weak ICA graphs/'
    #graph_dir = 'E:/Alex/Weak labelling test/Graphs'
    current_dir = None
    

    
    def __init__(self):
        pass
        

    def create_dir(self, name, session):
        dir_path = os.path.join(os.path.join(self.graph_dir,name),str(session))
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path,exist_ok=True)
        self.current_dir = os.path.abspath(dir_path)
        return self

    def plot_space(self, pos_bags, neg_bags):
        if not self.check_dim(pos_bags):
            pos_bags, neg_bags = self.get_manifold_projection(pos_bags, neg_bags)
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        label_info_pos = self.plot_bags(pos_bags, ax, color = 'r')
        label_info_neg = self.plot_bags(neg_bags, ax, color = 'g')

     
    def plot_bags(self, bags, ax, color = None):

        for i in range(len(bags)-1):
            self.plot_bag(bags[i], ax, color = color)
            self.plot_bag_outline(bags[i], ax)
            
            
        final = self.plot_bag(bags[-1], ax, color = color)
        return final
            


    def plot_bag(self,bag, ax, color = None):
        
        bag = np.scatter(bag)

        bag_plot = ax.scatter(bag[:,0], bag[:,1], color = color)
        
        return bag_plot
    

    def plot_bag_outline(self, bag, ax):
        
        mean = np.mean(bag, axis = 0)
        
        w = np.abs(np.max(bag[0]) - np.min(bag[0]))
        h = np.abs(np.max(bag[1]) - np.min(bag[1]))
        
        a = self.get_bag_angle(bag)
        
        outl = matplotlib.patches.Ellipse(mean, w,h)
        
    def get_bag_angle(self, bag):
        
        comp = np.max(bag, axis = 0) - np.min(bag, axis = 0)
        
        return np.cos([1,0]/ comp)
        

    def check_dim(self, bags):  
        return bags[0][0].shape[1] == 2
    
        
    def get_manifold_projection(self, pos_bags, neg_bags):
        
        all_points = self.combine_list([pos_bags,neg_bags])
        reducers = umap.UMAP()
        reducers.fit_transform(np.stack(all_points, axis = 0))
        sb = self.split_bags(all_points, [pos_bags,neg_bags])
        return sb[0], sb[1]

    
    def combine_list(self, bags):
        new_bags = []
        for b in bags:
            new_bags.extend(b)
        return new_bags

    
    def split_bags(self, all_bags:[np.ndarray], origin_bags:[list]):
        splits = []
        for i in range(len(origin_bags)):
            splits.append(len(origin_bags[i]))
            
        splits.append(len(all_bags))
            
        split_bags = []
        point = 0
        
        for i in range(len(splits)):
            split_bags.append(all_bags[point:splits[i]])
            point = splits[i]
            
        for i in range(len(split_bags)):
            split_bags[i] = self.split_class(split_bags[i], origin_bags[i])
            
        return split_bags
    
            

        
    def split_class(self,bag_list:[np.ndarray], orgin_bag):
        
        split_bag = []
        point = 0
        
        for b in orgin_bag:
            split_bag.append([])
            for i in b:
                split_bag[-1].append(bag_list[point])
                point += 1
                
        return split_bag
                
            
        
            
