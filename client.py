#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:43:56 2021

@author: krishna
"""
class Client:
    
    def __init__(self, dataset_x, dataset_y, epoch_number, learning_rate,model,weights,batch):
        self.dataset_x=dataset_x
        self.dataset_y=dataset_y
        self.epoch_number=epoch_number
        self.learning_rate=learning_rate
        self.weights=weights
        self.batch=batch
        self.model = model
        self.model.set_weights(self.weights)
        self.model.compile(loss='sparse_categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
        
    def train(self): 
        import numpy as np
        import pandas as pd
        import matplotlib as plt
        from tensorflow import keras
        import server
        
        history=self.model.fit(self.dataset_x, self.dataset_y,epochs=self.epoch_number,batch_size=self.batch) 
        
        output_weight=self.model.get_weights()

        return output_weight
        
        
        

        



    