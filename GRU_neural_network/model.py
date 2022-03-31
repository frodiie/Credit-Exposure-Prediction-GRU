"""
@Authors: Viktor Sambergs & Isabelle Frodé
@Date: Feb 2021
"""


import numpy as np
import string


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, Bidirectional
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint



def build_model(layers: int, nodes: list, learning_rate:float, loss_weights: list, 
                dropout: np.ndarray, regularizaion: string, _input: int, _feats: int, _output: int):
    model = Sequential()
    for layer in range(0,layers-2):            
        model.add(Bidirectional(GRU(nodes[layer], activation = 'tanh', return_sequences = True), 
                                input_shape = (_input, _feats)))
        if (dropout is not None) and (dropout):
            model.add(Dropout(dropout.pop(0)))
    model.add(Bidirectional(GRU(nodes[layers[layer+1]], activation = 'tanh'), 
                            input_shape = (_input, _feats)))
    model.add(Dense(_output))
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse', loss_weights=loss_weights)
    return model
