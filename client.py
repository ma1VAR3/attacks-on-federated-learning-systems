#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:43:56 2021

@author: krishna
"""
class Client:
    
    def __init__(self, dataset_x, dataset_y, epoch_number, learning_rate,weights,batch):
        self.dataset_x=dataset_x
        self.dataset_y=dataset_y
        self.epoch_number=epoch_number
        self.learning_rate=learning_rate
        self.weights=weights
        self.batch=batch
        
    def train(self): 
        import numpy as np
        import pandas as pd
        import matplotlib as plt
        from tensorflow import keras
        import server
        

        model=keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)),
            keras.layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu"),

            keras.layers.MaxPooling2D(pool_size=(2,2)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation="relu"),
            keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation="relu"),

            keras.layers.MaxPooling2D(pool_size=(2,2)),
            keras.layers.BatchNormalization(),  
            keras.layers.Conv2D(filters=256, kernel_size = (3,3), activation="relu"),
                
            keras.layers.MaxPooling2D(pool_size=(2,2)),
                
            keras.layers.Flatten(),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(512,activation="relu"),
                
            keras.layers.Dense(10,activation="softmax")
        ])


        model.set_weights(self.weights)
        
        model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
        history=model.fit(self.dataset_x, self.dataset_y,epochs=self.epoch_number,batch_size=self.batch,steps_per_epoch = self.dataset_x.shape[0] // self.batch) 
        
        output_weight=model.get_weights()

        return output_weight
        
        
        

        



    