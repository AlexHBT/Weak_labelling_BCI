import tensorflow as tf
import keras

from keras import layers, Sequential
from keras import Model

import tensorflow as tf

from sklearn.model_selection import train_test_split

class Mlp():
    
    model = None
    
    def __init__(self):
        
        pass
        
   

    
    def classify_fold_accuracy(self, X, y):
        
        X, X_test, y, y_test = train_test_split(
                                            X, y, test_size=0.2, 
                                            random_state=42)
        
        self.create_model(X.shape[1])
        
        return self.classify(X,y, X_test, y_test)       

    
    def classify(self, X,y, X_test, y_test):
        
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            patience=10,
            mode='auto',
            baseline=None,
            restore_best_weights=True)
        

        X,X_val, y, y_val = train_test_split(
                        X, y, test_size=0.2, 
                        random_state=467)
        
        history = self.model.fit(
        X,y,
        validation_data=(X_val, y_val),
        batch_size = 1,
        epochs = 100,
        callbacks=[callback],
        verbose=2
        )
        
        metrics = self.model.evaluate(X_test,y_test)
        print(metrics)
        return metrics[1]
        

    def create_model(self, input_shape: int):
        
        self.model = keras.Sequential([

            layers.Dense(400, activation='relu', input_shape=[input_shape]),
            layers.Dense(300, activation='relu'),  
            layers.Dense(200, activation='relu'),
            layers.Dense(100, activation='relu'),  
            layers.Dense(3, activation='sigmoid'),
    
        ])

        self.model.compile(
            optimizer = 'adam',
            loss = keras.losses.SparseCategoricalCrossentropy(
                from_logits=False,
                ignore_class=None,
                reduction='sum_over_batch_size',
                name='sparse_categorical_crossentropy'
                ),
            metrics=[keras.metrics.SparseCategoricalAccuracy()]
            
            )
        
    def evaluate(self,X,y):
        return self.model.evaluate(X,y)[1]