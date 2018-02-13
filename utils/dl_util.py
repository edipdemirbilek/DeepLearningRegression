# -*- coding: utf-8 -*-
"""
@author: edip.demirbilek

Deep Learning Util.

This module allow us to create/train/test and log evrything about Deep Learning
based models.
"""
import random
import time
import os

from statistics import mean
import math

from numpy import power
from numpy.random import uniform

from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
from keras import regularizers

from utils.dataset_util import prepare_data, unpack_partitioned_data, \
    substitute_training_data
from utils.common_util import accumulate_results_from_folds, compute_metrics, \
    compute_and_accumulate_results_from_counts

# File Names
DL_RESULTS_DETAILS_FILE_NAME = "dl_details.txt"
DL_RESULTS_SUMMARY_FILE_NAME = "dl_summary.csv"


def pack_regularization(dropout, k_l2, k_l1, a_l2, a_l1):
    """
    Packs dropout, kernel and activation regularization settings.

    Arguments:
        dropout -- add dropout layer, boolean
        k_l2 -- set kernel L2 regularization, boolean
        k_l1 -- set kernel L1 regularization, boolean
        a_l2 -- set activation L2 regularization, boolean
        a_l1 -- set activation L1 regularization, boolean

    Returns:
        regularization -- Dictionary of regularization settings
    """
    regularization = {}
    regularization["dropout"] = dropout
    regularization["k_l2"] = k_l2
    regularization["k_l1"] = k_l1
    regularization["a_l2"] = a_l2
    regularization["a_l1"] = a_l1
    return regularization


def unpack_regularization(regularization):
    """
    Unpacks dropout, kernel and activation regularization settings.

    Arguments:
        regularization -- Dictionary of regularization settings

    Returns:
        dropout -- add dropout layer, boolean
        k_l2 -- set kernel L2 regularization, boolean
        k_l1 -- set kernel L1 regularization, boolean
        a_l2 -- set activation L2 regularization, boolean
        a_l1 -- set activation L1 regularization, boolean
    """
    if regularization is None:
        return None, None, None, None, None

    dropout = regularization["dropout"]
    k_l2 = regularization["k_l2"]
    k_l1 = regularization["k_l1"]
    a_l2 = regularization["a_l2"]
    a_l1 = regularization["a_l1"]
    return dropout, k_l2, k_l1, a_l2, a_l1


def pack_model_hyperparameters(f_type, n_features, n_layers, n_epoch, n_batch,
                               loss, optimizer):
    """
    Packs dl model hyperparameters.

    Arguments:
        f_type -- type of features used
        n_features - number of features
        n_layers -- number of hidden layers
        n_epoch -- number of epochs
        n_batch -- batch size
        loss -- loss function
        optimizer -- optimizer

    Returns:
        hyperparameters -- Dictionary of model hyperparameters
    """
    hyperparameters = {}
    hyperparameters["f_type"] = f_type
    hyperparameters["n_features"] = n_features
    hyperparameters["n_layers"] = n_layers
    hyperparameters["n_epoch"] = n_epoch
    hyperparameters["n_batch"] = n_batch
    hyperparameters["loss"] = loss
    hyperparameters["optimizer"] = optimizer
    return hyperparameters


def unpack_model_hyperparameters(hyperparameters):
    """
    Unpacks dl model hyperparameters.

    Arguments:
        hyperparameters -- Dictionary of model hyperparameters

    Returns:
        f_type -- type of features used
        n_features - number of features
        n_layers -- number of hidden layers
        n_epoch -- number of epochs
        n_batch -- batch size
        loss -- loss function
        optimizer -- optimizer
    """
    f_type = hyperparameters["f_type"]
    n_features = hyperparameters["n_features"]
    n_layers = hyperparameters["n_layers"]
    n_epoch = hyperparameters["n_epoch"]
    n_batch = hyperparameters["n_batch"]
    loss = hyperparameters["loss"]
    optimizer = hyperparameters["optimizer"]
    return f_type, n_features, n_layers, n_epoch, n_batch, loss, optimizer


def pack_layer_hyperparameters(h_activation, o_activation, kernel_initializer):
    """
    Packs dl layer hyperparameters.

    Arguments:
        h_activation -- hidden layer activation function
        o_activation -- output layer activation function
        kernel_initializer -- kernel initializer

    Returns:
        hyperparameters -- Dictionary of dl layer hyperparameters
    """
    hyperparameters = {}
    hyperparameters["h_activation"] = h_activation
    hyperparameters["o_activation"] = o_activation
    hyperparameters["kernel_initializer"] = kernel_initializer
    return hyperparameters


def unpack_layer_hyperparameters(hyperparameters):
    """
    Unpacks dl layer hyperparameters.

    Arguments:
        hyperparameters -- Dictionary of dl layer hyperparameters

    Returns:
        h_activation -- hidden layer activation function
        o_activation -- output layer activation function
        kernel_initializer -- kernel initializer
    """
    h_activation = hyperparameters["h_activation"]
    o_activation = hyperparameters["o_activation"]
    kernel_initializer = hyperparameters["kernel_initializer"]
    return h_activation, o_activation, kernel_initializer


def initialize_layer_parameters(regularization):
    """
    Initializes kernal and activation regularization based on the settings in
    regularization object..

    Arguments:
        regularization -- Dictionary of shape:
            {
                'dropout': dropout,
                'k_l2': k_l2,
                'k_l1': k_l1,
                'a_l2': a_l2,
                'a_l1': a_l1
            }

    Returns:
        k_regularizer -- kernel regularizer
        a_regularizer -- activation regularizer
        k_v -- kernel regularizer value, float
        a_v -- activation regularizer value, float
    """
    _, k_l2, k_l1, a_l2, a_l1 = unpack_regularization(regularization)
    k_regularizer = None
    a_regularizer = None
    k_v = power(10, -1 * uniform(1, 4))
    a_v = power(10, -1 * uniform(1, 4))

    if k_l2 and k_l1:
        k_regularizer = regularizers.l1_l2(k_v)
    elif k_l2:
        k_regularizer = regularizers.l2(k_v)
    elif k_l1:
        k_regularizer = regularizers.l1(k_v)
    else:
        k_regularizer = None
        k_v = 0.

    if a_l2 and a_l1:
        a_regularizer = regularizers.l1_l2(a_v)
    elif a_l2:
        a_regularizer = regularizers.l2(a_v)
    elif a_l1:
        a_regularizer = regularizers.l1(a_v)
    else:
        a_regularizer = None
        a_v = 0.

    return k_regularizer, a_regularizer, k_v, a_v


def log_layer_parameters(layer, n_nodes, l_hyperparameters, regularization,
                         rate, k_v, a_v):
    """
    Logs hidden layer parameters to DL_RESULTS_DETAILS_FILE_NAME  file and
    stdout.

    Arguments:
        layer -- hidden layer index, int
        l_hyperparameters -- layer hyper parameters
        regularization -- dropout, kernel and activation regularization
            settings
        rate -- droout date, float
        k_v -- kernel regularization value, float
        a_v -- activation regularization value, float

    Returns:
        None
    """
    h_activation, o_activation, kernel_initializer = \
        unpack_layer_hyperparameters(l_hyperparameters)

    dropout, k_l2, k_l1, a_l2, a_l1 = unpack_regularization(regularization)

    with open(DL_RESULTS_DETAILS_FILE_NAME, "a") as file:
        if layer != 'output' and dropout:
            file.write("\n    dropout: " + str(dropout))
            print("    dropout: " + str(dropout))
            file.write("\n    rate: " + str(rate))
            print("    rate: " + str(rate))

        file.write("\n    layer: " + str(layer))
        print("    layer: " + str(layer))
        file.write("\n    n_nodes: " + str(n_nodes))
        print("    n_nodes: " + str(n_nodes))

        if layer == 'output':
            file.write("\n    o_activation: " + str(o_activation))
            print("    o_activation: " + str(o_activation))
        else:
            file.write("\n    h_activation: " + str(h_activation))
            print("    h_activation: " + str(h_activation))

            if layer == 1:
                file.write("\n    kernel_initializer: " + str(kernel_initializer))
                print("    kernel_initializer: " + str(kernel_initializer))

            if k_l2:
                file.write("\n    k_l2: " + str(k_l2))
                print("    k_l2: " + str(k_l2))
            if k_l1:
                file.write("\n    k_l1: " + str(k_l1))
                print("    k_l1: " + str(k_l1))
            if k_l2 or k_l1:
                file.write("\n    k_v: " + str(k_v))
                print("    k_v: " + str(k_v))
            if a_l2:
                file.write("\n    a_l2: " + str(a_l2))
                print("    a_l2: " + str(a_l2))
            if a_l1:
                file.write("\n    a_l1: " + str(a_l1))
                print("    a_l1: " + str(a_l1))
            if a_l2 or a_l1:
                file.write("\n    a_v: " + str(a_v))
                print("    a_v: " + str(a_v))


def log_model_hyperparameters(test_id, m_hyperparameters, regularization):
    """
    Logs model's hyperparameters to DL_RESULTS_DETAILS_FILE_NAME file and
    stdout.

    Arguments:
        test_id -- test id, string
        m_hyperparameters -- model hyper parameters
        regularization -- dropout, kernel and activation regularization
            settings

    Returns:
        None
    """
    f_type, n_features, n_layers, n_epoch, n_batch, loss, optimizer = \
        unpack_model_hyperparameters(m_hyperparameters)

    dropout, k_l2, k_l1, a_l2, a_l1 \
        = unpack_regularization(regularization)

    log_string \
        = "\nTest Id: {}, Feature Type: {}, Num Features: {}, Num Layers: {}, \
        Num Epochs: {}, Num Batch Size: {}, Dropout: {}, \
        k_l2: {}, k_l1: {}, a_l2: {}, a_l1: {}"\
        .format(test_id, f_type, n_features, n_layers, n_epoch, n_batch,
                dropout, k_l2, k_l1, a_l2, a_l1)
    print(log_string)

    with open(DL_RESULTS_DETAILS_FILE_NAME, "a") as file:
        file.write(log_string)


def add_dl_layers(dl_model, m_hyperparameters, l_hyperparameters,
                  regularization):
    """
    Adds hidden layers to the input model using settings provided.

    Arguments:
        dl_model -- deep learning model instance of type
            keras.models.Sequential
        m_hyperparameters -- model hyper parameters
        l_hyperparameters -- layer hyper parameters
        regularization -- dropout, kernel and activation regularization
            settings

    Returns:
        None
    """
    # unpack hyperparameters
    f_type, n_features, n_layers, n_epoch, n_batch, loss, optimizer = \
        unpack_model_hyperparameters(m_hyperparameters)

    h_activation, o_activation, kernel_initializer = \
        unpack_layer_hyperparameters(l_hyperparameters)

    # define actual regularization settings
    o_dropout, orig_k_l2, orig_k_l1, orig_a_l2, orig_a_l1 \
        = unpack_regularization(regularization)
    k_l2 = orig_k_l2 & random.choice([True, False])
    k_l1 = orig_k_l1 & random.choice([True, False])
    a_l2 = orig_a_l2 & random.choice([True, False])
    a_l1 = orig_a_l1 & random.choice([True, False])

    n_nodes_per_hidden_layer = []
    # number of nodes per layer is between 2 and n_features
    for _ in range(0, n_layers):
        n_node = int(power(2, math.sqrt(n_features) * uniform(0, 0.8)))
        n_node = n_node if n_node > 2 else 2
        n_nodes_per_hidden_layer.append(n_node)

    upper_limit = 1.0

    regularization = pack_regularization(False, k_l2, k_l1, a_l2, a_l1)
    k_regularizer, a_regularizer, k_v, a_v \
        = initialize_layer_parameters(regularization)
    n_nodes = sorted(n_nodes_per_hidden_layer, reverse=True)[0]

    dl_model.add(Dense(
        n_nodes, input_dim=n_features, activation=h_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=k_regularizer,
        activity_regularizer=a_regularizer))

    log_layer_parameters(
        layer=1, n_nodes=n_nodes, l_hyperparameters=l_hyperparameters,
        regularization=regularization, rate=0, k_v=k_v, a_v=a_v)

    for i in range(1, n_layers):
        dropout = o_dropout & random.choice([True, False])
        rate = uniform(0, upper_limit)
        if dropout:
            upper_limit /= 2
            dl_model.add(Dropout(rate, noise_shape=None, seed=None))

        n_nodes = sorted(n_nodes_per_hidden_layer, reverse=True)[i]
        k_l2 = orig_k_l2 & random.choice([True, False])
        k_l1 = orig_k_l1 & random.choice([True, False])
        a_l2 = orig_a_l2 & random.choice([True, False])
        a_l1 = orig_a_l1 & random.choice([True, False])

        regularization = pack_regularization(
            dropout, k_l2, k_l1, a_l2, a_l1)
        k_regularizer, a_regularizer, k_v, a_v \
            = initialize_layer_parameters(regularization)

        dl_model.add(Dense(
            n_nodes, activation=h_activation,
            kernel_regularizer=k_regularizer,
            activity_regularizer=a_regularizer))

        log_layer_parameters(
            layer=i+1, n_nodes=n_nodes, l_hyperparameters=l_hyperparameters,
            regularization=regularization, rate=rate, k_v=k_v, a_v=a_v)


def create_dl_model(m_hyperparameters, l_hyperparameters, regularization):
    """
    Creates a deep learning model using settings provided.

    Arguments:
        m_hyperparameters -- model hyperparameters
        l_hyperparameters -- layer hyperparameters
        regularization -- dropout, kernel and activation regularization
            settings

    Returns:
        dl_model -- deep learning model
    """
    f_type, n_features, n_layers, n_epoch, n_batch, loss, optimizer = \
        unpack_model_hyperparameters(m_hyperparameters)

    h_activation, o_activation, kernel_initializer = \
        unpack_layer_hyperparameters(l_hyperparameters)

    dl_model = Sequential()

    add_dl_layers(
        dl_model, m_hyperparameters, l_hyperparameters, regularization)

    dl_model.add(Dense(1, activation=o_activation))
    log_layer_parameters(layer='output', n_nodes=1,
                         l_hyperparameters=l_hyperparameters,
                         regularization=None, rate=None, k_v=None, a_v=None)

    dl_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return dl_model


def train_dl_model(dl_model, x_train, y_train, m_hyperparameters):
    """
    Trains the deep learning model.

    Arguments:
        dl_model -- deep learning model of type keras.models.Sequential
        x_train -- training attributes(data) of type numpy.ndarray and shape
            (training_data_size, n_features)
        y_train -- training labels of type numpy.ndarray and shape
            (training_data_size, )
        m_hyperparameters -- model hyperparameters

    Returns:
        history -- model training history of type keras.callbacks.History
    """
    f_type, n_features, n_layers, n_epoch, n_batch, loss, optimizer = \
        unpack_model_hyperparameters(m_hyperparameters)

    history = dl_model.fit(
        x_train, y_train, batch_size=n_batch, epochs=n_epoch, verbose=0)

    return history


def save_dl_header():
    """
    Saves CSV file header to DL_RESULTS_SUMMARY_FILE_NAME file.

    Arguments:
        None

    Returns:
        None
    """
    if not os.path.exists(DL_RESULTS_SUMMARY_FILE_NAME):
        with open(DL_RESULTS_SUMMARY_FILE_NAME, "a") as f:
            f.write("\ntest_id, f_type, n_features, n_layers, n_epoch, \
                    n_batch, rmse, rmse_epsilon, pearson, elapsed_time, \
                    dropout, k_l2, k_l1, a_l2, a_l1\n")


def save_dl_results(test_id, m_hyperparameters, regularization,
                    rmse_all_counts, rmse_epsilon_all_counts,
                    pearson_all_counts, elapsed_time):
    """
    Saves averaged results of running a deep learning model over k-fold cross
    validation and repeating 'count' times to DL_RESULTS_SUMMARY_FILE_NAME and
    DL_RESULTS_DETAILS_FILE_NAME file.

    Arguments:
        test_id -- test id, string
        m_hyperparameters -- model hyperparameters
        regularization -- dropout, kernel and activation regularization
            settings
        rmse_all_counts -- root mean square error, list
        rmse_epsilon_all_counts -- epsilon root mean square error, list
        pearson_all_counts -- pearson correlation coefficient, list
        elapsed_time -- elapsed time, string

    Returns:
        None
    """
    f_type, n_features, n_layers, n_epoch, n_batch, loss, optimizer = \
        unpack_model_hyperparameters(m_hyperparameters)

    dropout, k_l2, k_l1, a_l2, a_l1 \
        = unpack_regularization(regularization)

    result_for_print \
        = "Test Id: {}, Feature Tyoe: {}, Num Features: {}, Num Layers: {}, \
Num Epochs: {}, Num Batch Size: {}, Dropout: {}, k_l2: {}, k_l1: {}, a_l2: \
{}, a_l1: {}, RMSE: {}, Epsilon RMSE: {}, Pearson: {}, Elapsed Time: {}"\
            .format(test_id, f_type, n_features, n_layers, n_epoch, n_batch,
                    dropout, k_l2, k_l1, a_l2, a_l1, mean(rmse_all_counts),
                    mean(rmse_epsilon_all_counts), mean(pearson_all_counts),
                    elapsed_time)
    print("\nOverall Results:\n" + result_for_print)

    result_string_for_csv = '\n{}, {},{},{},{},{},{},{},{},{},{},{},{},{},{}'\
        .format(test_id, f_type, n_features, n_layers, n_epoch, n_batch,
                mean(rmse_all_counts), mean(rmse_epsilon_all_counts),
                mean(pearson_all_counts), elapsed_time, dropout,
                k_l2, k_l1, a_l2, a_l1)

    with open(DL_RESULTS_SUMMARY_FILE_NAME, "a") as file:
        file.write(result_string_for_csv)

    with open(DL_RESULTS_DETAILS_FILE_NAME, "a") as file:
        file.write(result_string_for_csv)


def run_dl_model(attributes, labels, test_id, dl_model, count, k,
                 m_hyperparameters, l_hyperparameters, regularization,
                 verbose):
    """
    Runs deep learning model for given attributes/labels using the given
    hyperparameters and logs the results to files and stdout.

    Args:
        attributes -- attributes(data) to use for training and test
        labels -- data labels to use for training and test
        test_id -- test id, string
        dl_model -- deep learning model to train
        count -- repeat count
        k -- number of folds for k-fold crossvalidation
        m_hyperparameters -- model hyperparameters
        l_hyperparameters -- layer hyperparameters
        regularization -- dropout, kernel and activation regularization
            settings
        verbose -- verbose flag

    Returns:
        None
    """
    start_time = time.time()

    log_model_hyperparameters(test_id, m_hyperparameters, regularization)

    f_type, n_features, n_layers, n_epoch, n_batch, loss, optimizer = \
        unpack_model_hyperparameters(m_hyperparameters)

    h_activation, o_activation, kernel_initializer = \
        unpack_layer_hyperparameters(l_hyperparameters)

    rmse_all_counts = []
    rmse_epsilon_all_counts = []
    pearson_all_counts = []

    if dl_model is None:
        dl_model = create_dl_model(m_hyperparameters, l_hyperparameters,
                                   regularization)

    model_weights = dl_model.get_weights()

    if verbose:
        print("\nModel Weights:\n" + str(model_weights))

    for count in range(1, count+1):
        print("\nCount: " + str(count) + " Time: " + time.ctime())

        y_test_all_folds = []
        prediction_all_folds = []
        prediction_epsilon_all_folds = []

        i_fold = 0

        k_fold = KFold(n_splits=k)
        for train_index, test_index in k_fold.split(attributes):

            partitioned_data = prepare_data(
                attributes, train_index, test_index, labels, n_features,
                f_type)

            x_train, x_test, y_train, y_test, ci_high, ci_low \
                = unpack_partitioned_data(partitioned_data)

            dl_model.set_weights(model_weights)

            train_dl_model(dl_model, x_train, y_train, m_hyperparameters)

            prediction = dl_model.predict(x_test)

            y_test_all_folds, prediction_all_folds, \
                prediction_epsilon_all_folds, prediction_epsilon \
                = accumulate_results_from_folds(
                    y_test_all_folds, prediction_all_folds,
                    prediction_epsilon_all_folds, i_fold, y_test,
                    prediction, ci_high, ci_low)

            print("\nMetrics for fold: " + str(i_fold + 1))
            compute_metrics(y_test, prediction, prediction_epsilon,
                            verbose)

            i_fold += 1

        print("\nMetrics for count: " + str(count))

        rmse_all_counts, rmse_epsilon_all_counts, pearson_all_counts \
            = compute_and_accumulate_results_from_counts(
                    y_test_all_folds, prediction_all_folds,
                    prediction_epsilon_all_folds, rmse_all_counts,
                    rmse_epsilon_all_counts, pearson_all_counts, verbose)

    elapsed_time = time.strftime("%H:%M:%S",
                                 time.gmtime(time.time()-start_time))
    save_dl_results(test_id, m_hyperparameters, regularization,
                    rmse_all_counts, rmse_epsilon_all_counts,
                    pearson_all_counts, elapsed_time)
