import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import umap.umap_ as umap
import matplotlib.patches



class bag_graphs(object):
    pass


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

    def plot_space(self, pos_bags, neg_bags, name = ''):
        
        if not self.check_dim(pos_bags):
            pos_bags, neg_bags = self.get_manifold_projection(pos_bags, neg_bags)
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        label_info_pos = self.plot_bags(pos_bags, ax, color = 'r')
        label_info_neg = self.plot_bags(neg_bags, ax, color = 'g')
        ax.set_title(f'{name} Bag scatter graph with bags.png')
        ax.set_xlabel('UMAP feature 1')
        ax.set_ylabel('UMAP feature 2')
        path = os.path.join(self.current_dir,f'{name} vs bag scatter.png')
        fig.savefig(path)

     
    def plot_bags(self, bags, ax, color = None):

        for i in range(len(bags)-1):
            self.plot_bag(bags[i], ax, color = color)
            self.plot_bag_outline(bags[i], ax)
            
            
        final = self.plot_bag(bags[-1], ax, color = color)
        self.plot_bag_outline(bags[i], ax)
        return final
            


    def plot_bag(self,bag, ax, color = None):
        
        bag = np.stack(bag, axis = 0)

        bag_plot = ax.scatter(bag[:,0], bag[:,1], color = color)
        
        return bag_plot
    

    def plot_bag_outline(self, bag, ax):
        
        bag = np.stack(bag, axis = 0)
        
        mean = np.mean(bag, axis = 0)
        
        w = np.abs(np.max(bag[0]) - np.min(bag[0]))
        h = np.abs(np.max(bag[1]) - np.min(bag[1]))
        
        a = self.get_bag_angle(bag)
        
        outl = matplotlib.patches.Ellipse((mean[0],mean[1]), w,h, angle = a, fill = False)
        
        ax.add_patch(outl)
        
    def get_bag_angle(self, bag):
        
        comp = np.max(bag, axis = 0) - np.min(bag, axis = 0)
        
        return np.cos(1/ np.sum(comp))
        

    def check_dim(self, bags):  
        return bags[0][0].shape == 2
    
        
    def get_manifold_projection(self, pos_bags, neg_bags):
        
        all_points = np.concatenate(self.combine_list([pos_bags,neg_bags]), axis = 0)
        
        reducers = umap.UMAP()
        all_points = reducers.fit_transform(np.stack(all_points, axis = 0))
        sb = self.split_bags(all_points, [pos_bags,neg_bags])
        return sb[0], sb[1]

    
    def combine_list(self, bags):
        new_bags = []
        for b in bags:
            t = type(b[0])
            if type(b[0]).__name__ == 'list':
                new_bags.append(self.combine_list(b))
            else:
                new_bags.extend(b)
        return new_bags

    
    def split_bags(self, all_bags:[np.ndarray], origin_bags:[list]):
        splits = [0]
        for i in range(len(origin_bags)):
            for j in range(len(origin_bags[i])):
                splits.append(len(origin_bags[i][j])+splits[-1])
            
        #splits.append(len(all_bags))
            
        split_bags = []
        point = 0
        
        for i in range(len(splits)-1):
            split_bags.append(all_bags[point:splits[i+1]])
            point = splits[i+1]
         
        class_splits = self.split_class(split_bags, origin_bags)
          
        return class_splits
    
            

        
    def split_class(self,bag_list:[np.ndarray], orgin_bag):
        
        split_bag = []
        point = 0
        
        for b in orgin_bag:
            split_bag.append([])
            for i in b:
                split_bag[-1].append(self.sqeeze_bag(np.split(bag_list[0], bag_list[0].shape[0])))
                bag_list.pop(0)
                
        return split_bag

    def sqeeze_bag(self, bag):
        for i in range(len(bag)):
            bag[i] = np.squeeze(bag[i])
        return bag