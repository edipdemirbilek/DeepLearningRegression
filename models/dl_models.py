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
from keras.layers import Dropout
from keras.layers.core import Dense
from keras.optimizers import Adadelta, Adamax, Nadam, RMSprop, SGD
from keras import regularizers

from utils.dl_util import pack_regularization


def add_output_layer_and_compile(dl_model):
    """
    Arguments:
        dl_model -- Deep Learning model of type keras.models.Sequential

    Returns:
        dl_model -- Deep Learning model of type keras.models.Sequential
    """
    dl_model.add(Dense(1, activation='softplus'))
    adadelta = Adadelta()
    dl_model.compile(loss='mse', optimizer=adadelta, metrics=['accuracy'])

    return dl_model


def dl_model_1():
    """
    Test Id: 530.0989173665997305,
    Feature Type: sorted_subjects,
    Num Features: 9,
    Num Layers: 2,
    Num Epochs: 783,
    Num Batch Size: 120,
    Loss: binary_crossentropy,
    Optimizer: Nadam,
    Dropout: False, k_l2: False, k_l1: False, a_l2: False, a_l1: False

    Random Search Result
    test_id, f_type, n_features, n_layers, n_epoch, n_batch, rmse, rmse_epsilon, pearson, elapsed_time, dropout, k_l2, k_l1, a_l2, a_l1, Count
    530.0989173665997305, sorted_subjects,9,2,783,120,0.36392871695331613,0.21367863183310884,0.927363051328573,00:19:32,False,False,False,False,False

    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 2,
            h_activation: 'softsign',
            kernel_initializer: 'glorot_normal',
        Layer 2 -- Dense
            #nodes: 2
            activation: 'softsign',
        Layer 3 -- Dense, Output
            #nodes: 1
            activation: 'relu'

    Returns:
        dl_model -- Deep Learning model of type keras.models.Sequential
        n_features - Number of features, int
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
    print("Custom DL model 1")

    n_features = 9
    n_layers = 2
    n_epoch = 783
    n_batch_size = 120

    dropout = False
    k_l2 = False
    k_l1 = False
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    loss = 'binary_crossentropy'
    optimizer = Nadam()

    dl_model = Sequential()

    dl_model.add(Dense(2, input_dim=n_features, activation='softsign',
                       kernel_initializer='glorot_normal'))

    dl_model.add(Dense(2, activation='softsign'))

    dl_model.add(Dense(1, activation='relu'))
    dl_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return dl_model, n_features, n_layers, n_epoch, n_batch_size, \
        regularization, loss, optimizer

def dl_model_2():
    """

    Test Id: 250.03366711196429795,
    Feature Type: sorted_subjects,
    Num Features: 17,
    Num Layers: 2,
    Num Epochs: 412,
    Num Batch Size: 120,
    Loss: mean_absolute_error,
    Optimizer: <keras.optimizers.Adadelta object at 0x126343f28>,
    Dropout: False, k_l2: False, k_l1: False, a_l2: False, a_l1: False

    Random Search Result
    test_id, f_type, n_features, n_layers, n_epoch, n_batch, rmse, rmse_epsilon, pearson, elapsed_time, dropout, k_l2, k_l1, a_l2, a_l1, Count
    250.03366711196429795, sorted_subjects,17,2,412,120,0.3965292041268971,0.24324761507356363,0.9271592079243844,00:10:42,False,False,False,False,False

   Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 3,
            h_activation: 'softplus',
            kernel_initializer: 'orthogonal',
        Layer 2 -- Dense
            #nodes: 2
            activation: 'softplus',
        Layer 3 -- Dense, Output
            #nodes: 1
            activation: 'hard_sigmoid'

    Returns:
        dl_model -- Deep Learning model of type keras.models.Sequential
        n_features - Number of features, int
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
    print("Custom DL model 1")

    n_features = 17
    n_layers = 2
    n_epoch = 412
    n_batch_size = 120

    dropout = False
    k_l2 = False
    k_l1 = False
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    loss = 'mean_absolute_error'
    optimizer = Adadelta()

    dl_model = Sequential()

    dl_model.add(Dense(3, input_dim=n_features, activation='softplus',
                       kernel_initializer='orthogonal'))

    dl_model.add(Dense(2, activation='softplus'))

    dl_model.add(Dense(1, activation='hard_sigmoid'))
    dl_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return dl_model, n_features, n_layers, n_epoch, n_batch_size, \
        regularization, loss, optimizer

def dl_model_3():
    """

    Test Id: 110.7219142396918702,
    Feature Type: sorted_subjects,
    Num Features: 8,
    Num Layers: 2,
    Num Epochs: 749,
    Num Batch Size: 120,
    Loss: binary_crossentropy,
    Optimizer: <keras.optimizers.Adadelta object at 0x12a100940>,
    Dropout: False, k_l2: False, k_l1: False, a_l2: False, a_l1: False

    Random Search Result
    test_id, f_type, n_features, n_layers, n_epoch, n_batch, rmse, rmse_epsilon, pearson, elapsed_time, dropout, k_l2, k_l1, a_l2, a_l1, Count
    110.7219142396918702, sorted_subjects,8,2,749,120,0.367103318272342,0.21427375171379326,0.9268053958866134,00:14:50,False,False,False,False,False

    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 4,
            h_activation: 'relu',
            kernel_initializer: 'truncated_normal',
        Layer 2 -- Dense
            #nodes: 2
            activation: 'relu',
        Layer 3 -- Dense, Output
            #nodes: 1
            activation: 'sigmoid'

    Returns:
        dl_model -- Deep Learning model of type keras.models.Sequential
        n_features - Number of features, int
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
    print("Custom DL model 1")

    n_features = 8
    n_layers = 2
    n_epoch = 749
    n_batch_size = 120

    dropout = False
    k_l2 = False
    k_l1 = False
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    loss = 'binary_crossentropy'
    optimizer = Adadelta()

    dl_model = Sequential()

    dl_model.add(Dense(4, input_dim=n_features, activation='relu',
                       kernel_initializer='truncated_normal'))

    dl_model.add(Dense(2, activation='relu'))

    dl_model.add(Dense(1, activation='sigmoid'))
    dl_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return dl_model, n_features, n_layers, n_epoch, n_batch_size, \
        regularization, loss, optimizer

def dl_model_4():
    """
    Test Id: 180.5254900038280402,
    Feature Type: sorted_subjects,
    Num Features: 12,
    Num Layers: 2,
    Num Epochs: 415,
    Num Batch Size: 120,
    Loss: mean_squared_error,
    Optimizer: <keras.optimizers.RMSprop object at 0x130193fd0>,
    Dropout: False, k_l2: False, k_l1: False, a_l2: False, a_l1: False

    Random Search Result
    test_id, f_type, n_features, n_layers, n_epoch, n_batch, rmse, rmse_epsilon, pearson, elapsed_time, dropout, k_l2, k_l1, a_l2, a_l1, Count
    180.5254900038280402, sorted_subjects,12,2,415,120,0.371165772417391,0.22013048287370057,0.9249325717307784,00:08:52,False,False,False,False,False

    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 6,
            h_activation: 'sigmoid',
            kernel_initializer: 'orthogonal',
        Layer 2 -- Dense
            #nodes: 2
            activation: 'sigmoid',
        Layer 3 -- Dense, Output
            #nodes: 1
            activation: 'hard_sigmoid'

    Returns:
        dl_model -- Deep Learning model of type keras.models.Sequential
        n_features - Number of features, int
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
    print("Custom DL model 4")

    n_features = 12
    n_layers = 2
    n_epoch = 415
    n_batch_size = 120

    dropout = False
    k_l2 = False
    k_l1 = False
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    loss = 'mean_squared_error'
    optimizer = RMSprop()

    dl_model = Sequential()

    dl_model.add(Dense(6, input_dim=n_features, activation='sigmoid',
                       kernel_initializer='orthogonal'))

    dl_model.add(Dense(2, activation='sigmoid'))

    dl_model.add(Dense(1, activation='hard_sigmoid'))
    dl_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return dl_model, n_features, n_layers, n_epoch, n_batch_size, \
        regularization, loss, optimizer


def dl_model_5():
    """
    Test Id: 50.030917668555019318,
    Feature Type: sorted_subjects,
    Num Features: 16,
    Num Layers: 2,
    Num Epochs: 967,
    Num Batch Size: 120,
    Loss: mean_squared_error,
    Optimizer: <keras.optimizers.RMSprop object at 0x11a3bebe0>,
    Dropout: False, k_l2: False, k_l1: False, a_l2: False, a_l1: False

    Random Search Result
    test_id, f_type, n_features, n_layers, n_epoch, n_batch, rmse, rmse_epsilon, pearson, elapsed_time, dropout, k_l2, k_l1, a_l2, a_l1, Count
    50.030917668555019318, sorted_subjects,16,2,967,120,0.3573825052842691,0.20378077360461308,0.9309577204930928,01:03:59,False,False,False,False,False

    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 3,
            h_activation: 'sigmoid',
            kernel_initializer: 'orthogonal',
        Layer 2 -- Dense
            #nodes: 2
            activation: 'sigmoid',
        Layer 3 -- Dense, Output
            #nodes: 1
            activation: 'hard_sigmoid'

    Returns:
        dl_model -- Deep Learning model of type keras.models.Sequential
        n_features - Number of features, int
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
    print("Custom DL model 1")

    n_features = 16
    n_layers = 2
    n_epoch = 967
    n_batch_size = 120

    dropout = False
    k_l2 = False
    k_l1 = False
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    loss = 'mean_squared_error'
    optimizer = RMSprop()

    dl_model = Sequential()

    dl_model.add(Dense(3, input_dim=n_features, activation='sigmoid',
                       kernel_initializer='orthogonal'))

    dl_model.add(Dense(2, activation='sigmoid'))

    dl_model.add(Dense(1, activation='hard_sigmoid'))
    dl_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return dl_model, n_features, n_layers, n_epoch, n_batch_size, \
        regularization, loss, optimizer

def dl_model_6():
    """
    Test Id: 100.982894254673383,
    Feature Type: sorted_subjects,
    Num Features: 15,
    Num Layers: 2,
    Num Epochs: 508,
    Num Batch Size: 120,
    Loss: mean_squared_error,
    Optimizer: <keras.optimizers.RMSprop object at 0x1212fde80>,
    Dropout: False, k_l2: False, k_l1: False, a_l2: False, a_l1: False

    Random Search Result
    test_id, f_type, n_features, n_layers, n_epoch, n_batch, rmse, rmse_epsilon, pearson, elapsed_time, dropout, k_l2, k_l1, a_l2, a_l1, Count
    100.982894254673383, sorted_subjects,15,2,508,120,0.35972264439460905,0.20674280294135497,0.9304692512000209,00:39:51,False,False,False,False,False

    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 8,
            h_activation: 'sigmoid',
            kernel_initializer: 'glorot_normal',
        Layer 2 -- Dense
            #nodes: 4
            activation: 'sigmoid',
        Layer 3 -- Dense, Output
            #nodes: 1
            activation: 'hard_sigmoid'

    Returns:
        dl_model -- Deep Learning model of type keras.models.Sequential
        n_features - Number of features, int
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
    print("Custom DL model 1")

    n_features = 15
    n_layers = 2
    n_epoch = 508
    n_batch_size = 120

    dropout = False
    k_l2 = False
    k_l1 = False
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    loss = 'mean_squared_error'
    optimizer = RMSprop()

    dl_model = Sequential()

    dl_model.add(Dense(8, input_dim=n_features, activation='sigmoid',
                       kernel_initializer='glorot_normal'))

    dl_model.add(Dense(4, activation='sigmoid'))

    dl_model.add(Dense(1, activation='hard_sigmoid'))
    dl_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return dl_model, n_features, n_layers, n_epoch, n_batch_size, \
        regularization, loss, optimizer