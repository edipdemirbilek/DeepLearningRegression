# -*- coding: utf-8 -*-
"""
@author: edip.demirbilek

This module includes all best perfroming deep learning models that have fully
connected layers. Model's Parameters are  obtained using random search over
a large set of hyperparameters.

Todo:
    * Add all best performing models.
"""
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adadelta
from keras import regularizers

from deep_learning.dl_utils import pack_regularization_object


def create_model_1(n_features):
    """
    Model 1 -- Fully connected layers
        loss -- 'Mean Square Error'mse''
        optimizer -- Adadelta
        metrics -- ['accuracy']
    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 100,
            activation: 'tanh',
            kernel_initializer: 'uniform',
            kernel_regularizer: l1 value 0.000772747534144
        Layer 2 -- Dense
            #nodes: 54
            activation: 'tanh',
            kernel_regularizer: l1 value 0.00119620962974
        Layer 3 -- Dense
            #nodes: 4
            activation: 'tanh',
            kernel_regularizer: l1 value 0.000136272407271
        Layer 4 -- Dense, Output
            #nodes: 1
            activation: 'softplus'

    Arguments:
        n_features -- Number of features, int

    Returns:
        dl_model -- Deep Learning model of type keras.models.Sequential
        n_layers -- Number of layers, int value 3
        n_epoch -- Number og epochs, int value 644
        n_batch_size -- Batch size, int value 120
        regularization -- Dictionary of shape
            {
                'dropout': False,
                'k_l2': False,
                'k_l1': True,
                'a_l2': False,
                'a_l1': False,
            }
    """
    n_layers = 3
    n_epoch = 644
    n_batch_size = 120

    dropout = False
    k_l2 = False
    k_l1 = True
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization_object(
        dropout, k_l2, k_l1, a_l2, a_l1)

    dl_model = Sequential()

    dl_model.add(Dense(100, input_dim=n_features, activation='tanh',
                       kernel_initializer='uniform',
                       kernel_regularizer=regularizers.l1(0.000772747534144),
                       activity_regularizer=None))

    dl_model.add(Dense(54, activation='tanh',
                       kernel_regularizer=regularizers.l1(0.00119620962974),
                       activity_regularizer=None))
    dl_model.add(Dense(4, activation='tanh',
                       kernel_regularizer=regularizers.l1(0.000136272407271),
                       activity_regularizer=None))

    dl_model.add(Dense(1, activation='softplus'))
    adadelta = Adadelta()
    dl_model.compile(loss='mse', optimizer=adadelta, metrics=['accuracy'])

    return dl_model, n_layers, n_epoch, n_batch_size, regularization
