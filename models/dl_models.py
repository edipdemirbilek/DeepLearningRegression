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
from keras.optimizers import Adadelta
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


def dl_model_1(n_features):
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
    print("DL model 1")
    n_layers = 3
    n_epoch = 644
    n_batch_size = 120

    dropout = False
    k_l2 = False
    k_l1 = True
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
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

    return add_output_layer_and_compile(dl_model), n_layers, n_epoch, \
        n_batch_size, regularization


def dl_model_2(n_features):
    """
    Model 2 -- Fully connected layers
        loss -- 'Mean Square Error'mse''
        optimizer -- Adadelta
        metrics -- ['accuracy']
    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 93,
            activation: 'tanh',
            kernel_initializer: 'uniform',
            kernel_regularizer: l1 value 0.00291949389517
        Layer 2 -- Dense
            #nodes: 61
            activation: 'tanh',
            kernel_regularizer: l1 value 0.000558804570517
        Layer 3 -- Dense
            #nodes: 14
            activation: 'tanh',
            kernel_regularizer: l1 value 0.000814892122429
        Layer 5 -- Dense
            #nodes: 6
            activation: 'tanh',
            kernel_regularizer: l1 value 0.00418878188865
        Layer 5 -- Dense, Output
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
    print("DL model 2")
    n_layers = 4
    n_epoch = 934
    n_batch_size = 120

    dropout = False
    k_l2 = False
    k_l1 = True
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    dl_model = Sequential()

    dl_model.add(Dense(93, input_dim=n_features, activation='tanh',
                       kernel_initializer='uniform',
                       kernel_regularizer=regularizers.l1(0.00291949389517),
                       activity_regularizer=None))

    dl_model.add(Dense(61, activation='tanh',
                       kernel_regularizer=regularizers.l1(0.000558804570517),
                       activity_regularizer=None))
    dl_model.add(Dense(14, activation='tanh',
                       kernel_regularizer=regularizers.l1(0.000814892122429),
                       activity_regularizer=None))
    dl_model.add(Dense(6, activation='tanh',
                       kernel_regularizer=regularizers.l1(0.00418878188865),
                       activity_regularizer=None))

    return add_output_layer_and_compile(dl_model), n_layers, n_epoch, \
        n_batch_size, regularization


def dl_model_3(n_features):
    """
    Model 3 -- Fully connected layers
        loss -- 'Mean Square Error'mse''
        optimizer -- Adadelta
        metrics -- ['accuracy']
    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 10,
            activation: 'tanh',
            kernel_initializer: 'uniform',
            kernel_regularizer: l1 value 0.00212630915942
        Layer 2 -- Dense, Output
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
    print("DL model 3")
    n_layers = 1
    n_epoch = 4194
    n_batch_size = 120

    dropout = False
    k_l2 = False
    k_l1 = True
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    dl_model = Sequential()

    dl_model.add(Dense(10, input_dim=n_features, activation='tanh',
                       kernel_initializer='uniform',
                       kernel_regularizer=regularizers.l1(0.00212630915942),
                       activity_regularizer=None))

    return add_output_layer_and_compile(dl_model), n_layers, n_epoch, \
        n_batch_size, regularization


def dl_model_4(n_features):
    """
    Model 4 -- Fully connected layers
        loss -- 'Mean Square Error'mse''
        optimizer -- Adadelta
        metrics -- ['accuracy']
    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 108,
            activation: 'tanh',
            kernel_initializer: 'uniform',
            kernel_regularizer: l1 value 0.000598295679733
        Layer 2 -- Dense
            #nodes: 25
            activation: 'tanh',
            kernel_regularizer: l1 value 0.000519263706192
        Layer 3 -- Dense
            #nodes: 6
            activation: 'tanh',
            kernel_regularizer: l1 value 0.000358395014507
        Layer 4 -- Dense
            #nodes: 3
            activation: 'tanh',
            kernel_regularizer: l1 value 0.00208375309026
        Layer 5 -- Dense, Output
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
    print("DL model 4")
    n_layers = 4
    n_epoch = 664
    n_batch_size = 120

    dropout = False
    k_l2 = False
    k_l1 = True
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    dl_model = Sequential()

    dl_model.add(Dense(108, input_dim=n_features, activation='tanh',
                       kernel_initializer='uniform',
                       kernel_regularizer=regularizers.l1(0.000598295679733),
                       activity_regularizer=None))

    dl_model.add(Dense(25, activation='tanh',
                       kernel_regularizer=regularizers.l1(0.000519263706192),
                       activity_regularizer=None))
    dl_model.add(Dense(6, activation='tanh',
                       kernel_regularizer=regularizers.l1(0.000358395014507),
                       activity_regularizer=None))
    dl_model.add(Dense(3, activation='tanh',
                       kernel_regularizer=regularizers.l1(0.00208375309026),
                       activity_regularizer=None))

    return add_output_layer_and_compile(dl_model), n_layers, n_epoch, \
        n_batch_size, regularization


def dl_model_5(n_features):
    """
    Model 5 -- Fully connected layers
        loss -- 'Mean Square Error'mse''
        optimizer -- Adadelta
        metrics -- ['accuracy']
    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 15,
            activation: 'tanh',
            kernel_initializer: 'uniform',
            kernel_regularizer: l1 value 0.000146971236117
        Layer 2 -- Dense
            #nodes: 8
            activation: 'tanh',
            kernel_regularizer: None
        Layer 3 -- Dense
            #nodes: 8
            activation: 'tanh',
            kernel_regularizer: l1 value 0.0190282049554
        Layer 4 -- Dense
            #nodes: 7
            activation: 'tanh',
            kernel_regularizer: l1 value 0.000663034551678
        Layer 5 -- Dense
            #nodes: 6
            activation: 'tanh',
            kernel_regularizer: l1 value None
            dropout: rate value 0.516280788566578
        Layer 6 -- Dense, Output
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
    print("DL model 5")
    n_layers = 5
    n_epoch = 4365
    n_batch_size = 120

    dropout = True
    k_l2 = False
    k_l1 = True
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    dl_model = Sequential()

    dl_model.add(Dense(15, input_dim=n_features, activation='tanh',
                       kernel_initializer='uniform',
                       kernel_regularizer=regularizers.l1(0.000146971236117),
                       activity_regularizer=None))

    dl_model.add(Dense(8, activation='tanh',
                       kernel_regularizer=None,
                       activity_regularizer=None))
    dl_model.add(Dense(8, activation='tanh',
                       kernel_regularizer=regularizers.l1(0.0190282049554),
                       activity_regularizer=None))
    dl_model.add(Dense(7, activation='tanh',
                       kernel_regularizer=regularizers.l1(0.000663034551678),
                       activity_regularizer=None))
    dl_model.add(Dropout(rate=0.516280788566578, noise_shape=None, seed=None))
    dl_model.add(Dense(6, activation='tanh',
                       kernel_regularizer=None,
                       activity_regularizer=None))

    return add_output_layer_and_compile(dl_model), n_layers, n_epoch, \
        n_batch_size, regularization


def dl_model_6(n_features):
    """
    Model 6 -- Fully connected layers
        loss -- 'Mean Square Error'mse''
        optimizer -- Adadelta
        metrics -- ['accuracy']
    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 12,
            activation: 'tanh',
            kernel_initializer: 'uniform',
            kernel_regularizer: l1 value 0.012871345521
        Layer 2 -- Dense
            #nodes: 6
            activation: 'tanh',
            kernel_regularizer: l1 value 0.00200697504526
        Layer 3 -- Dense, Output
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
    print("DL model 6")
    n_layers = 2
    n_epoch = 4905
    n_batch_size = 120

    dropout = False
    k_l2 = False
    k_l1 = True
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    dl_model = Sequential()

    dl_model.add(Dense(12, input_dim=n_features, activation='tanh',
                       kernel_initializer='uniform',
                       kernel_regularizer=regularizers.l1(0.012871345521),
                       activity_regularizer=None))

    dl_model.add(Dense(6, activation='tanh',
                       kernel_regularizer=regularizers.l1(0.00200697504526),
                       activity_regularizer=None))

    return add_output_layer_and_compile(dl_model), n_layers, n_epoch, \
        n_batch_size, regularization


def dl_model_7(n_features):
    """
    Model 7 -- Fully connected layers
        loss -- 'Mean Square Error'mse''
        optimizer -- Adadelta
        metrics -- ['accuracy']
    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 3,
            activation: 'tanh',
            kernel_initializer: 'uniform',
            kernel_regularizer: l1 value 0.00137064637093
        Layer 2 -- Dense
            #nodes: 3
            activation: 'tanh',
            kernel_regularizer: l1 value 0.000118085542081
        Layer 3 -- Dense
            #nodes: 2
            activation: 'tanh',
            kernel_regularizer: l1 value 0.00234570316304
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
    print("DL model 7")
    n_layers = 3
    n_epoch = 4441
    n_batch_size = 120

    dropout = False
    k_l2 = False
    k_l1 = True
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    dl_model = Sequential()

    dl_model.add(Dense(3, input_dim=n_features, activation='tanh',
                       kernel_initializer='uniform',
                       kernel_regularizer=regularizers.l1(0.00137064637093),
                       activity_regularizer=None))

    dl_model.add(Dense(3, activation='tanh',
                       kernel_regularizer=regularizers.l1(0.000118085542081),
                       activity_regularizer=None))

    dl_model.add(Dense(2, activation='tanh',
                       kernel_regularizer=regularizers.l1(0.00234570316304),
                       activity_regularizer=None))

    return add_output_layer_and_compile(dl_model), n_layers, n_epoch, \
        n_batch_size, regularization


def dl_model_8(n_features):
    """
    Model 8 -- Fully connected layers
        loss -- 'Mean Square Error'mse''
        optimizer -- Adadelta
        metrics -- ['accuracy']
    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 32,
            activation: 'tanh',
            kernel_initializer: 'uniform',
            kernel_regularizer: l1_l2 value 0.00282481903628
        Layer 2 -- Dense, Output
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
    print("DL model 8")
    n_layers = 1
    n_epoch = 7756
    n_batch_size = 120

    dropout = False
    k_l2 = True
    k_l1 = True
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    dl_model = Sequential()

    dl_model.add(Dense(32, input_dim=n_features, activation='tanh',
                       kernel_initializer='uniform',
                       kernel_regularizer=regularizers.l1_l2(0.00282481903628),
                       activity_regularizer=None))

    return add_output_layer_and_compile(dl_model), n_layers, n_epoch, \
        n_batch_size, regularization


def dl_model_9(n_features):
    """
    Model 9 -- Fully connected layers
        loss -- 'Mean Square Error'mse''
        optimizer -- Adadelta
        metrics -- ['accuracy']
    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 55,
            activation: 'tanh',
            kernel_initializer: 'uniform',
            kernel_regularizer: l2 value 0.00114660582694
        Layer 2 -- Dense
            #nodes: 9,
            activation: 'tanh',
            kernel_initializer: 'uniform',
            kernel_regularizer: l2 value 0.000193253640832
        Layer 3 -- Dense, Output
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
    print("DL model 9")
    n_layers = 2
    n_epoch = 3847
    n_batch_size = 120

    dropout = False
    k_l2 = True
    k_l1 = False
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    dl_model = Sequential()

    dl_model.add(Dense(55, input_dim=n_features, activation='tanh',
                       kernel_initializer='uniform',
                       kernel_regularizer=regularizers.l2(0.00114660582694),
                       activity_regularizer=None))

    dl_model.add(Dense(9, input_dim=n_features, activation='tanh',
                       kernel_initializer='uniform',
                       kernel_regularizer=regularizers.l2(0.000193253640832),
                       activity_regularizer=None))

    return add_output_layer_and_compile(dl_model), n_layers, n_epoch, \
        n_batch_size, regularization


def dl_model_10(n_features):
    """
    Model 10 -- Fully connected layers
        loss -- 'Mean Square Error'mse''
        optimizer -- Adadelta
        metrics -- ['accuracy']
    Layers
        Layer 0 -- Input Layer
            #nodes: n_features
        Layer 1 -- Dense
            #nodes: 51,
            activation: 'tanh',
            kernel_initializer: 'uniform',
            kernel_regularizer: l2 value 0.000447236290898
        Layer 2 -- Dense, Output
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
    print("DL model 10")
    n_layers = 1
    n_epoch = 2309
    n_batch_size = 120

    dropout = False
    k_l2 = True
    k_l1 = False
    a_l2 = False
    a_l1 = False
    regularization = pack_regularization(
        dropout, k_l2, k_l1, a_l2, a_l1)

    dl_model = Sequential()

    dl_model.add(Dense(51, input_dim=n_features, activation='tanh',
                       kernel_initializer='uniform',
                       kernel_regularizer=regularizers.l2(0.000447236290898),
                       activity_regularizer=None))

    return add_output_layer_and_compile(dl_model), n_layers, n_epoch, \
        n_batch_size, regularization
