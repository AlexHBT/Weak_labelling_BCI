import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping
import sklearn as sk
from sklearn.model_selection import KFold
import keras

class eegnet():
    dtype = None
    
    def __init__(self):
        pass
        
    def process_and_classify(self, data):
        data = self.check_dtype(data)
            
        X, y = self.convert_to_ml_data(data)
        
        return self.classify(X,y)
    
    def check_dtype(self, data):
        if self.dtype == 'inst':
            new_data = []
            for i in data:
                new_data.append(self.combine_bags(i.get_bags()).get_bag())
            
        else:
            return data

    def convert_to_ml_data(self,bags):
        y = []
        total_data = []
        for b in range(len(bags)):
            y.append(np.zeros(len(bags[b])) + b)
            for i in bags[b]:
                total_data.append(i)
        total_data = np.stack(total_data, axis = 0)
        return total_data, np.concatenate(y)



    def train_with_eeg_net(self, X,y, X_val, y_val, X_test, y_test):
        


        F1 = 2
        D = 1
        F2 = F1*D
        model = tf.keras.Sequential([
                tf.keras.layers.Input(shape = (16,250,1)),
                #tf.keras.layers.Reshape((1,16,750)),
                tf.keras.layers.Conv2D(F1,(1,64),activation = 'linear', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.DepthwiseConv2D((3,4), activation= 'linear', depth_multiplier = D,
                                                depthwise_constraint = tf.keras.constraints.MaxNorm(max_value=1),padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.AveragePooling2D((1,4)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.SeparableConv2D(F2, (5,3), activation = 'linear', padding = 'same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('elu'),
                tf.keras.layers.AveragePooling2D((1,8)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
        
                tf.keras.layers.Dense(1, activation = 'sigmoid',
                                      kernel_constraint = tf.keras.constraints.MaxNorm(max_value=0.25))
                ])

        model.compile(
            optimizer=keras.optimizers.Adam(
                    learning_rate=0.00001),
          loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
          metrics=['accuracy'])
    
        callback = EarlyStopping(
            monitor = 'val_accuracy',
            min_delta=0.1,
            patience=10,
            restore_best_weights=True,
        )
    
        history = model.fit(np.swapaxes(X,1,2)[...,np.newaxis],y,
                            validation_data = (np.swapaxes(X_val,1,2)[...,np.newaxis], y_val),
                            epochs = 100,
                            batch_size= X.shape[0],
                            verbose = 0,
                            callbacks= [callback])
        
        return model.evaluate(np.swapaxes(X_test,1,2)[...,np.newaxis], y_test, return_dict=True)['accuracy']


    def get_kfold_data_set(self, X,y):
        #kfold = KFold(5,shuffle=False)
        values = []
        for i in enumerate(KFold(5,shuffle=False).split(X,y)):
            values.append(i[1])
            
       
        return values
        
    def classify(self, X,y):
        indxs = self.get_kfold_data_set(X,y)
        
        accuracies = []
        for i in range(len(indxs)):
            X_train = X[indxs[i][0],:]
            y_train = y[indxs[i][0]]
            X_test = X[indxs[i][1],:]
            y_test = [indxs[i][1]]
            
            X_train, y_train, X_val, y_val = self.get_val_set(X_train,y_train)
            
            accuracies.append(self.train_with_eeg_net(X_train, y_train, 
                                                      X_val, y_val,
                                                      X_test, y_test))
            
        return np.mean(accuracies)
            
    def get_val_set(self, X,y, split = 20):
       
        indxs = self.get_kfold_data_set(X,y)
        X_train = X[indxs[0][0],:]
        y_train = y[indxs[0][0]]
        X_val = X[indxs[0][1],:]
        y_val = [indxs[0][1]]
        
        return X_train, y_train, X_val, y_val
        
        