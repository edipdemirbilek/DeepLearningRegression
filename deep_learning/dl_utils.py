# -*- coding: utf-8 -*-
"""
@author: edip.demirbilek

Dataset Utils..

This module allow us to read the Parametric and Bitstream version of the  INRS
Audiovisual Quality Dataset from file system.

Todo:
    * Read parametric version of the INRS Audiovisual Quality Dataset.
    * Complete Docstrings
"""
import random
import time

from statistics import mean

from numpy import array, power, zeros, concatenate
from numpy.random import uniform
import scipy as sp

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.optimizers import Adadelta
from keras import regularizers

from dataset_utils import prepare_data

# File Names
RESULTS_DETAILS_FILE_NAME = "Details_new.txt"
RESULTS_SUMMARY_FILE_NAME = "Results_new.csv"


def unpack_regularization_object(regularization):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    dropout = regularization["dropout"]
    k_l2 = regularization["k_l2"]
    k_l1 = regularization["k_l1"]
    a_l2 = regularization["a_l2"]
    a_l1 = regularization["a_l1"]
    return dropout, k_l2, k_l1, a_l2, a_l1


def pack_regularization_object(dropout, k_l2, k_l1, a_l2, a_l1):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    regularization = {}
    regularization["dropout"] = dropout
    regularization["k_l2"] = k_l2
    regularization["k_l1"] = k_l1
    regularization["a_l2"] = a_l2
    regularization["a_l1"] = a_l1
    return regularization


def initialize_layer_parameters(regularization):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    _, k_l2, k_l1, a_l2, a_l1 = unpack_regularization_object(regularization)
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


def log_layer_parameters(layer, n_hidden, regularization, rate, k_v, a_v):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    dropout, k_l2, k_l1, a_l2, a_l1 \
        = unpack_regularization_object(regularization)
    with open(RESULTS_DETAILS_FILE_NAME, "a") as file:
        if dropout:
            file.write("\n    dropout: " + str(dropout))
            file.write("\n    rate: " + str(rate))
            print("    dropout: " + str(dropout))
            print("    rate: " + str(rate))

        file.write("\n    layer: " + str(layer))
        file.write("\n    n_hidden: " + str(n_hidden))
        print("    layer: " + str(layer))
        print("    n_hidden: " + str(n_hidden))

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


def log_hyperparameters(test_id, n_features, n_layers, n_epoch, n_batch_size,
                        regularization):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    dropout, k_l2, k_l1, a_l2, a_l1 \
        = unpack_regularization_object(regularization)
    log_string \
        = "\nTest Id: {}, Num Features: {}, Num Layers: {}, Num Epochs: {}, \
        Num Batch Size: {}, Dropout: {}, \
        k_l2: {}, k_l1: {}, a_l2: {}, a_l1: {}"\
        .format(test_id, n_features, n_layers, n_epoch, n_batch_size, dropout,
                k_l2, k_l1, a_l2, a_l1)
    print(log_string)

    with open(RESULTS_DETAILS_FILE_NAME, "a") as file:
        file.write(log_string)


def add_layers(dl_model, n_features, n_layers, regularization):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    o_dropout, orig_k_l2, orig_k_l1, orig_a_l2, orig_a_l1 \
        = unpack_regularization_object(regularization)

    k_l2 = orig_k_l2 & random.choice([True, False])
    k_l1 = orig_k_l1 & random.choice([True, False])
    a_l2 = orig_a_l2 & random.choice([True, False])
    a_l1 = orig_a_l1 & random.choice([True, False])

    n_nodes_per_hidden_layer = []
    for _ in range(0, n_layers):
        n_nodes_per_hidden_layer.append(
            int(power(2, 7 * uniform(0.145, 1.0))))

    upper_limit = 1.0

    regularization = pack_regularization_object(False, k_l2, k_l1, a_l2, a_l1)
    k_regularizer, a_regularizer, k_v, a_v \
        = initialize_layer_parameters(regularization)
    n_hidden = sorted(n_nodes_per_hidden_layer, reverse=True)[0]

    dl_model.add(Dense(
        n_hidden, input_dim=n_features, activation='tanh',
        kernel_initializer='uniform',
        kernel_regularizer=k_regularizer,
        activity_regularizer=a_regularizer))

    log_layer_parameters(
        1, n_hidden, regularization, 0, k_v, a_v)

    for i in range(1, n_layers):

        dropout = o_dropout & random.choice([True, False])

        rate = uniform(0, upper_limit)

        if dropout:
            upper_limit /= 2
            dl_model.add(Dropout(rate, noise_shape=None, seed=None))

        n_hidden = sorted(n_nodes_per_hidden_layer, reverse=True)[i]
        k_l2 = orig_k_l2 & random.choice([True, False])
        k_l1 = orig_k_l1 & random.choice([True, False])
        a_l2 = orig_a_l2 & random.choice([True, False])
        a_l1 = orig_a_l1 & random.choice([True, False])

        regularization = pack_regularization_object(
            dropout, k_l2, k_l1, a_l2, a_l1)
        k_regularizer, a_regularizer, k_v, a_v \
            = initialize_layer_parameters(regularization)

        dl_model.add(Dense(
            n_hidden, activation='tanh',
            kernel_regularizer=k_regularizer,
            activity_regularizer=a_regularizer))

        log_layer_parameters(
            i+1, n_hidden, regularization, rate, k_v, a_v)


def create_model(n_layers, n_features, regularization):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    dl_model = Sequential()

    add_layers(
        dl_model, n_features, n_layers, regularization)

    dl_model.add(Dense(1, activation='softplus'))
    adadelta = Adadelta()
    dl_model.compile(loss='mse', optimizer=adadelta, metrics=['accuracy'])

    return dl_model


def train_model(dl_model, x_train, y_train, n_batch_size, n_epoch, x_test,
                y_test):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    history = dl_model.fit(
        x_train, y_train, batch_size=n_batch_size, epochs=n_epoch,
        verbose=0, validation_data=(x_test, y_test))
    return history


def save_results(test_id, n_features, n_layers, n_epoch, n_batch_size,
                 regularization, rmse_per_count, rmse_epsilon_per_count,
                 pearson_per_count, elapsed_time):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    dropout, k_l2, k_l1, a_l2, a_l1 \
        = unpack_regularization_object(regularization)
    result_for_print \
        = "Test Id: {}, Num Features: {}, Num Layers: {}, Num Epochs: {}, \
        Num Batch Size: {}, Dropout: {}, k_l2: {}, k_l1: {}, a_l2: {}, \
        a_l1: {}, RMSE: {}, Epsilon RMSE: {}, Pearson: {}, Elapsed Time: {}"\
            .format(test_id, n_features, n_layers, n_epoch, n_batch_size,
                    dropout, k_l2, k_l1, a_l2, a_l1, mean(rmse_per_count),
                    mean(rmse_epsilon_per_count), mean(pearson_per_count),
                    elapsed_time)
    print("\nOverall Results:\n" + result_for_print)

    result_string_for_csv = '\n{},{},{},{},{},{},{},{},{},{},{},{},{},{}'\
        .format(test_id, n_features, n_layers, n_epoch, n_batch_size,
                mean(rmse_per_count), mean(rmse_epsilon_per_count),
                mean(pearson_per_count), elapsed_time, dropout,
                k_l2, k_l1, a_l2, a_l1)

    with open(RESULTS_SUMMARY_FILE_NAME, "a") as file:
        file.write(result_string_for_csv)

    with open(RESULTS_DETAILS_FILE_NAME, "a") as file:
        file.write(result_string_for_csv)


def accumulate_results_from_folds(y_test_for_all_folds, prediction_folds,
                                  prediction_epsilon_folds, i_fold,
                                  prediction_from_fold, ci_low, ci_high,
                                  y_test):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    prediction_epsilon = zeros(len(prediction_from_fold))

    for index in range(0, len(prediction_from_fold)):
        if prediction_from_fold[index] < ci_low[index]:
            prediction_epsilon[index] \
                = y_test[index]-(prediction_from_fold[index]-ci_low[index])
        elif prediction_from_fold[index] > ci_high[index]:
            prediction_epsilon[index] \
                = y_test[index]+(ci_high[index]-prediction_from_fold[index])
        else:
            prediction_epsilon[index] = y_test[index]

    if i_fold == 0:
        y_test_for_all_folds = y_test[:]
        prediction_folds = prediction_from_fold[:].tolist()
        prediction_epsilon_folds = prediction_epsilon[:].tolist()
    else:
        y_test_for_all_folds = concatenate([y_test_for_all_folds, y_test])
        prediction_folds = concatenate(
            [prediction_folds, prediction_from_fold[:]])
        prediction_epsilon_folds = concatenate(
            [prediction_epsilon_folds, prediction_epsilon[:]])

    return y_test_for_all_folds, prediction_folds,\
        prediction_epsilon_folds, prediction_epsilon


def compute_metrics(y_test_normalized, prediction_normalized,
                    prediction_epsilon_normalized, debug):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    y_test = y_test_normalized * 5
    prediction = prediction_normalized * 5
    prediction_epsilon = prediction_epsilon_normalized * 5

    if debug:
        print("            y_test Normalized: " +
              ', '.join(["%.2f" % e for e in y_test_normalized]))
        print("        Prediction Normalized: " +
              ', '.join(["%.2f" % e for e in prediction_normalized]))
        print("Epsilon Prediction Normalized: " +
              ', '.join(["%.2f" % e for e in prediction_epsilon_normalized]))

    if debug:
        print("                       y_test: " +
              ', '.join(["%.2f" % e for e in y_test]))
        print("                   Prediction: " +
              ', '.join(["%.2f" % e for e in prediction]))
        print("           Epsilon Prediction: " +
              ', '.join(["%.2f" % e for e in prediction_epsilon]))

    # mse=mean_squared_error(y_test, prediction_from_fold)
    # rmse https://www.kaggle.com/wiki/RootMeanSquaredError

    rmse = mean_squared_error(y_test, prediction)**0.5
    rmse_epsilon = mean_squared_error(y_test, prediction_epsilon)**0.5

    # converting to a one dimensional array here
    prediction = [arr[0] for arr in prediction]

    r_value, p_value = sp.stats.pearsonr(array(y_test), array(prediction))

    if debug:
        print("        RMSE: %.3f" % rmse)
        print("Epsilon RMSE: %.3f" % rmse_epsilon)
        print("     Pearson: %.3f" % r_value)

    return rmse, rmse_epsilon, r_value, p_value


def compute_results(y_test_for_all_folds, prediction_folds,
                    prediction_epsilon_folds, rmse_per_count,
                    rmse_epsilon_per_count, pearson_per_count, debug):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    rmse, rmse_epsilon, r_value, _ = compute_metrics(
        y_test_for_all_folds, prediction_folds,
        prediction_epsilon_folds, debug)

    rmse_per_count.append(rmse)
    rmse_epsilon_per_count.append(rmse_epsilon)
    pearson_per_count.append(r_value)

    return rmse_per_count, rmse_epsilon_per_count, pearson_per_count


def run_model(attributes, labels, test_id, dl_model, count, k, n_features,
              n_layers, n_epoch, n_batch_size, regularization, debug):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    start_time = time.time()

    log_hyperparameters(
        test_id, n_features, n_layers, n_epoch, n_batch_size, regularization)

    rmse_per_count = []
    rmse_epsilon_per_count = []
    pearson_per_count = []

    if dl_model is None:
        dl_model = create_model(n_layers, n_features, regularization)

    model_weights = dl_model.get_weights()

    if debug:
        print("\nModel Weights:\n" + str(model_weights))

    for count in range(1, count+1):
        print("\nCount: " + str(count) + " Time: " + time.ctime())

        y_test_for_all_folds = []
        prediction_folds = []
        prediction_epsilon_folds = []
        i_fold = 0

        k_fold = KFold(n_splits=k)
        for train_index, test_index in k_fold.split(attributes):
            x_train, y_train, x_test, y_test, ci_low, ci_high = prepare_data(
                attributes, train_index, test_index, labels, n_features)
            dl_model.set_weights(model_weights)
            train_model(dl_model, x_train, y_train, n_batch_size, n_epoch,
                        x_test, y_test)
            prediction_from_fold = dl_model.predict(x_test)

            y_test_for_all_folds, prediction_folds, \
                prediction_epsilon_folds, prediction_epsilon \
                = accumulate_results_from_folds(
                    y_test_for_all_folds, prediction_folds,
                    prediction_epsilon_folds, i_fold,
                    prediction_from_fold, ci_low, ci_high, y_test)

            if debug:
                print("\nMetrics for fold: " + str(i_fold + 1))
                compute_metrics(y_test, prediction_from_fold,
                                prediction_epsilon, debug)

            i_fold += 1

        if debug:
            print("\nMetrics for count: " + str(count))
        rmse_per_count, rmse_epsilon_per_count, pearson_per_count \
            = compute_results(y_test_for_all_folds, prediction_folds,
                              prediction_epsilon_folds, rmse_per_count,
                              rmse_epsilon_per_count, pearson_per_count, debug)

    elapsed_time = time.strftime("%H:%M:%S",
                                 time.gmtime(time.time()-start_time))
    save_results(
        test_id, n_features, n_layers, n_epoch, n_batch_size, regularization,
        rmse_per_count, rmse_epsilon_per_count, pearson_per_count,
        elapsed_time)
