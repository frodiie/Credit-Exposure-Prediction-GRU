"""
@Authors: Viktor Sambergs & Isabelle Frodé
@Date: Mars 2022
"""

import keras
import numpy as np
import string


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


def build_model(layers: int, units: list, learning_rate: float, 
                _input: int, _feats: int, _output: int):
    """
    Returns GRU network architecture.

    ARGS:
        layers (int):                     nbr of layers.
        units (list):                     nbr of units/layer.
        learning_rate (float):            Adam learning rate
        _input (int):                     input sequence length.
        _feats (int):                     nbr of features.
        _output (int):                    output sequence length.
    """
    model = Sequential()
    for layer in range(0,layers-2):            
        model.add(GRU(units[layer], 
                      activation = 'tanh', 
                      return_sequences = True, 
                      input_shape = (_input, _feats)))
    model.add(GRU(units[layers[layer+1]], 
                  activation = 'tanh', 
                  input_shape = (_input, _feats)))
    model.add(Dense(_output))
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse')
    return model


def train_model(train_x: np.ndarray, train_y: np.ndarray, 
                model: keras.engine.sequential.Sequential):
    """
    Returns trained model and training history.
    
    ARGS:
        train_x (np.ndarray):             input data.
        train_y:                          target data.
        model (Sequential):               model architecture.
    """
    checkpoint = ModelCheckpoint("best_model.hdf5", 
                                 monitor='val_loss', 
                                 verbose=1,
                                 save_best_only=True, 
                                 mode='auto', 
                                 period=1)
    
    history = model.fit(train_x, 
                        train_y, 
                        epochs = 100, 
                        batch_size = 32, 
                        validation_split = 0.2, 
                        verbose = 1, 
                        shuffle = True, 
                        callbacks = [checkpoint])
    return model, history
