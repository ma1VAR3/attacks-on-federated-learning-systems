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
        keras.layers.Flatten(input_shape=[784,]),
        keras.layers.Dense(256,activation='tanh'),
        keras.layers.Dense(128,activation='tanh'),
        keras.layers.Dense(10,activation='softmax')
        ])


        model.set_weights(self.weights)
        
        model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=self.learning_rate),metrics=['accuracy'])
        history=model.fit(self.dataset_x, self.dataset_y,epochs=self.epoch_number,batch_size=self.batch) 
        
        output_weight=model.get_weights()

        return output_weight
        
        
        

        



    