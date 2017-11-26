#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:17:37 2017

@author: edip.demirbilek
"""
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adadelta
from keras import regularizers

from deep_learning.dl_utils import pack_regularization_object


def create_model_1(n_features):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

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
