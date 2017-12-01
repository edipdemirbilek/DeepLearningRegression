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

import os

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

from dataset_utils import prepare_data, unpack_partitioned_data

# File Names
DL_RESULTS_DETAILS_FILE_NAME = "dl_details.txt"
DL_RESULTS_SUMMARY_FILE_NAME = "dl_summary.csv"


def pack_regularization_object(dropout, k_l2, k_l1, a_l2, a_l1):
    """
    Packs dropout, kernel and activation regularization settings.

    Arguments:
        dropout -- add dropout layer, boolean
        k_l2 -- set kernel L2 regularization, boolean
        k_l1 -- set kernel L1 regularization, boolean
        a_l2 -- set activation L2 regularization, boolean
        a_l1 -- set activation L1 regularization, boolean

    Returns:
        regularization -- Dictionary of shape:
            {
                'dropout': dropout,
                'k_l2': k_l2,
                'k_l1': k_l1,
                'a_l2': a_l2,
                'a_l1': a_l1
            }
    """
    regularization = {}
    regularization["dropout"] = dropout
    regularization["k_l2"] = k_l2
    regularization["k_l1"] = k_l1
    regularization["a_l2"] = a_l2
    regularization["a_l1"] = a_l1
    return regularization


def unpack_regularization_object(regularization):
    """
    Unpacks dropout, kernel and activation regularization settings.

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
        dropout -- add dropout layer, boolean
        k_l2 -- set kernel L2 regularization, boolean
        k_l1 -- set kernel L1 regularization, boolean
        a_l2 -- set activation L2 regularization, boolean
        a_l1 -- set activation L1 regularization, boolean
    """
    dropout = regularization["dropout"]
    k_l2 = regularization["k_l2"]
    k_l1 = regularization["k_l1"]
    a_l2 = regularization["a_l2"]
    a_l1 = regularization["a_l1"]
    return dropout, k_l2, k_l1, a_l2, a_l1


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


def log_layer_parameters(layer, n_nodes, regularization, rate, k_v, a_v):
    """
    Logs hidden layer parameters to DL_RESULTS_DETAILS_FILE_NAME  file and
    stdout.

    Arguments:
        layer -- hidden layer index, int
        n_nodes -- number of nodes, int
        regularization -- dropout, kernel and activation regularization
            settings
        rate -- droout date, float
        k_v -- kernel regularization value, float
        a_v -- activation regularization value, float

    Returns:
        None
    """
    dropout, k_l2, k_l1, a_l2, a_l1 \
        = unpack_regularization_object(regularization)
    with open(DL_RESULTS_DETAILS_FILE_NAME, "a") as file:
        if dropout:
            file.write("\n    dropout: " + str(dropout))
            file.write("\n    rate: " + str(rate))
            print("    dropout: " + str(dropout))
            print("    rate: " + str(rate))

        file.write("\n    layer: " + str(layer))
        file.write("\n    n_nodes: " + str(n_nodes))
        print("    layer: " + str(layer))
        print("    n_nodes: " + str(n_nodes))

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


def log_hyperparameters(test_id, n_features, n_layers, n_epoch, n_batch,
                        regularization):
    """
    Logs model's hyperparameters to DL_RESULTS_DETAILS_FILE_NAME file and
    stdout.

    Arguments:
        test_id -- test id, string
        n_features -- number of features, int
        n_layers -- number of hidden layers, int
        n_epoch -- number of epochs, int
        n_batch -- batch size, int
        regularization -- dropout, kernel and activation regularization
            settings

    Returns:
        None
    """
    dropout, k_l2, k_l1, a_l2, a_l1 \
        = unpack_regularization_object(regularization)
    log_string \
        = "\nTest Id: {}, Num Features: {}, Num Layers: {}, Num Epochs: {},\
        Num Batch Size: {}, Dropout: {}, \
        k_l2: {}, k_l1: {}, a_l2: {}, a_l1: {}"\
        .format(test_id, n_features, n_layers, n_epoch, n_batch, dropout,
                k_l2, k_l1, a_l2, a_l1)
    print(log_string)

    with open(DL_RESULTS_DETAILS_FILE_NAME, "a") as file:
        file.write(log_string)


def add_layers(dl_model, n_features, n_layers, regularization):
    """
    Adds hidden layers to the input model using settings provided.

    Arguments:
        dl_model -- deep learning model instance of type
            keras.models.Sequential
        n_features -- number of inout features, int
        n_layers -- number of hidden layers, int
        regularization -- dropout, kernel and activation regularization
            settings

    Returns:
        None
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
    n_nodes = sorted(n_nodes_per_hidden_layer, reverse=True)[0]

    dl_model.add(Dense(
        n_nodes, input_dim=n_features, activation='tanh',
        kernel_initializer='uniform',
        kernel_regularizer=k_regularizer,
        activity_regularizer=a_regularizer))

    log_layer_parameters(
        1, n_nodes, regularization, 0, k_v, a_v)

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

        regularization = pack_regularization_object(
            dropout, k_l2, k_l1, a_l2, a_l1)
        k_regularizer, a_regularizer, k_v, a_v \
            = initialize_layer_parameters(regularization)

        dl_model.add(Dense(
            n_nodes, activation='tanh',
            kernel_regularizer=k_regularizer,
            activity_regularizer=a_regularizer))

        log_layer_parameters(
            i+1, n_nodes, regularization, rate, k_v, a_v)


def create_model(n_layers, n_features, regularization):
    """
    Creates a deep learning model using settings provided.

    Arguments:
        n_layers -- number of hidden layers, int
        n_features -- number of input features, int
        regularization -- dropout, kernel and activation regularization
            settings

    Returns:
        dl_model -- deep learning model
    """
    dl_model = Sequential()

    add_layers(
        dl_model, n_features, n_layers, regularization)

    dl_model.add(Dense(1, activation='softplus'))
    adadelta = Adadelta()
    dl_model.compile(loss='mse', optimizer=adadelta, metrics=['accuracy'])

    return dl_model


def train_model(dl_model, x_train, y_train, n_batch, n_epoch):
    """
    Trains the deep learning model.

    Arguments:
        dl_model -- deep learning model of type keras.models.Sequential
        x_train -- training attributes(data) of type numpy.ndarray and shape
            (training_data_size, n_features)
        y_train -- training labels of type numpy.ndarray and shape
            (training_data_size, )
        n_batch -- batch size, int
        n_epoch -- number of epochs, int

    Returns:
        history -- model training history of type keras.callbacks.History
    """
    history = dl_model.fit(
        x_train, y_train, batch_size=n_batch, epochs=n_epoch, verbose=0)
    return history


def save_header():
    """
    Saves CSV file header to DL_RESULTS_SUMMARY_FILE_NAME file.

    Arguments:
        None

    Returns:
        None
    """
    if not os.path.exists(DL_RESULTS_SUMMARY_FILE_NAME):
        with open(DL_RESULTS_SUMMARY_FILE_NAME, "a") as f:
            f.write("\ntest_id, n_features, n_layers, n_epoch, n_batch, \
                    rmse, rmse_epsilon, pearson, elapsed_time, dropout, \
                    k_l2, k_l1, a_l2, a_l1\n")


def save_results(test_id, n_features, n_layers, n_epoch, n_batch,
                 regularization, rmse, rmse_epsilon,
                 pearson, elapsed_time):
    """
    Saves averaged results of running a deep learning model over k-fold cross
    validation and repeating 'count' times to DL_RESULTS_SUMMARY_FILE_NAME and
    DL_RESULTS_DETAILS_FILE_NAME file.

    Arguments:
        test_id -- test id, string
        n_features -- number of features, int
        n_layers -- number of hidden layers, int
        n_epoch -- number of epochs, int
        n_batch -- batch size, int
        regularization -- dropout, kernel and activation regularization
            settings
        rmse -- root mean square error, list
        rmse_epsilon -- epsilon root mean square error, list
        pearson -- pearson correlation coefficient, list
        elapsed_time -- elapsed time, string

    Returns:
        None
    """
    dropout, k_l2, k_l1, a_l2, a_l1 \
        = unpack_regularization_object(regularization)
    result_for_print \
        = "Test Id: {}, Num Features: {}, Num Layers: {}, Num Epochs: {}, \
Num Batch Size: {}, Dropout: {}, k_l2: {}, k_l1: {}, a_l2: {}, \
a_l1: {}, RMSE: {}, Epsilon RMSE: {}, Pearson: {}, Elapsed Time: {}"\
            .format(test_id, n_features, n_layers, n_epoch, n_batch,
                    dropout, k_l2, k_l1, a_l2, a_l1, mean(rmse),
                    mean(rmse_epsilon), mean(pearson),
                    elapsed_time)
    print("\nOverall Results:\n" + result_for_print)

    result_string_for_csv = '\n{},{},{},{},{},{},{},{},{},{},{},{},{},{}'\
        .format(test_id, n_features, n_layers, n_epoch, n_batch,
                mean(rmse), mean(rmse_epsilon),
                mean(pearson), elapsed_time, dropout,
                k_l2, k_l1, a_l2, a_l1)

    with open(DL_RESULTS_SUMMARY_FILE_NAME, "a") as file:
        file.write(result_string_for_csv)

    with open(DL_RESULTS_DETAILS_FILE_NAME, "a") as file:
        file.write(result_string_for_csv)


def accumulate_results_from_folds(y_test_all_folds, prediction_all_folds,
                                  prediction_epsilon_all_folds, i_fold,
                                  y_test, prediction, ci_high, ci_low):
    """
    Adds results from the current fold to the overall k-fold cross
    validation results.

    Arguments:
        y_test_all_folds -- placeholder to store test labels from all folds
        prediction_all_folds -- placeholder to store prediction labels from all
            folds
        prediction_epsilon_all_folds -- placeholder to store epsilon prediction
            labels from all folds
        i_fold -- fold index, int
        y_test - test labels from current fold
        prediction -- prediction labels from current fold
        ci_high -- 95% confidence interval high values for the labels from
            current fold
        ci_low -- 95% confidence interval low values for the labels from
            current fold

    Returns:
        y_test_all_folds -- placeholder to store test labels from all folds
        prediction_all_folds -- placeholder to store prediction labels from all
            folds
        prediction_epsilon_all_folds -- placeholder to store epsilon prediction
            labels from all folds
        prediction_epsilon -- epsilon prediction labels from current fold
    """
    prediction_epsilon = zeros(len(prediction))

    for index in range(0, len(prediction)):
        if prediction[index] < ci_low[index]:
            prediction_epsilon[index] \
                = y_test[index]-(prediction[index]-ci_low[index])
        elif prediction[index] > ci_high[index]:
            prediction_epsilon[index] \
                = y_test[index]+(ci_high[index]-prediction[index])
        else:
            prediction_epsilon[index] = y_test[index]

    if i_fold == 0:
        y_test_all_folds = y_test[:]
        prediction_all_folds = prediction[:].tolist()
        prediction_epsilon_all_folds = prediction_epsilon[:].tolist()
    else:
        y_test_all_folds = concatenate([y_test_all_folds, y_test])
        prediction_all_folds = concatenate(
            [prediction_all_folds, prediction[:]])
        prediction_epsilon_all_folds = concatenate(
            [prediction_epsilon_all_folds, prediction_epsilon[:]])

    return y_test_all_folds, prediction_all_folds,\
        prediction_epsilon_all_folds, prediction_epsilon


def compute_metrics(y_test_normalized, prediction_normalized,
                    prediction_epsilon_normalized, verbose):
    """
    Computes RMSE, Epsilon RMSE and Pearson correlation coefficients.

    Argument:
        y_test_normalized -- normalized test labels
        prediction_normalized -- normalized prediction labels
        prediction_epsilon_normalized -- normalized epsilon prediction labels
        verbose -- verbose flag

    Returns:
        rmse -- root mean square value, float
        rmse_epsilon -- epsilon rmse value, float
        r_value -- pearson's correlation coefficient'
        p_value -- 2-tailed p-value
    """
    y_test = y_test_normalized * 5
    prediction = prediction_normalized * 5
    prediction_epsilon = prediction_epsilon_normalized * 5

    if verbose:
        print("            y_test Normalized: " +
              ', '.join(["%.2f" % e for e in y_test_normalized]))
        print("        Prediction Normalized: " +
              ', '.join(["%.2f" % e for e in prediction_normalized]))
        print("Epsilon Prediction Normalized: " +
              ', '.join(["%.2f" % e for e in prediction_epsilon_normalized]))

    if verbose:
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

    print("        RMSE: %.3f" % rmse)
    print("Epsilon RMSE: %.3f" % rmse_epsilon)
    print("     Pearson: %.3f" % r_value)

    return rmse, rmse_epsilon, r_value, p_value


def compute_results(y_test_all_folds, prediction_all_folds,
                    prediction_epsilon_all_folds, rmse_all_counts,
                    rmse_epsilon_all_counts, pearson_all_counts, verbose):
    """
    Computes rmse, epsilon rmse and pearson correlation for all folds and
    stores the result to the placeholders for all counts.

    Args:
        y_test_all_folds -- test labels from all folds
        prediction_all_folds -- prediction labels from all folds
        prediction_epsilon_all_folds -- epsilon prediction labels from all
            folds
        rmse_all_counts -- placeholder to store rmse value for all counts
        rmse_epsilon_all_counts -- placeholder to store epsilon rmse value for
            all counts
        pearson_all_counts -- placeholder to store pearson correlation for all
            counts
        verbose -- verbose flag


    Returns:
        rmse_all_counts -- placeholder to store rmse value for all counts
        rmse_epsilon_all_counts -- placeholder to store epsilon rmse value for
            all counts
        pearson_all_counts -- placeholder to store pearson correlation for all
            counts
    """
    rmse, rmse_epsilon, r_value, _ = compute_metrics(
        y_test_all_folds, prediction_all_folds,
        prediction_epsilon_all_folds, verbose)

    rmse_all_counts.append(rmse)
    rmse_epsilon_all_counts.append(rmse_epsilon)
    pearson_all_counts.append(r_value)

    return rmse_all_counts, rmse_epsilon_all_counts, pearson_all_counts


def run_model(attributes, labels, test_id, dl_model, count, k, n_features,
              n_layers, n_epoch, n_batch, regularization, verbose):
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
        n_features -- number of inout features, int
        n_layers -- number of hidden layers, int
        n_epoch -- number of epocs, int
        n_batch -- batch size, int
        regularization -- dropout, kernel and activation regularization
            settings
        verbose -- verbose flag

    Returns:
        None
    """
    start_time = time.time()

    log_hyperparameters(
        test_id, n_features, n_layers, n_epoch, n_batch, regularization)

    rmse_all_counts = []
    rmse_epsilon_all_counts = []
    pearson_all_counts = []

    if dl_model is None:
        dl_model = create_model(n_layers, n_features, regularization)

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
                attributes, train_index, test_index, labels, n_features)

            x_train, x_test, y_train, y_test, ci_high, ci_low \
                = unpack_partitioned_data(partitioned_data)

            dl_model.set_weights(model_weights)

            train_model(dl_model, x_train, y_train, n_batch, n_epoch)

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
            = compute_results(y_test_all_folds, prediction_all_folds,
                              prediction_epsilon_all_folds, rmse_all_counts,
                              rmse_epsilon_all_counts, pearson_all_counts,
                              verbose)

    elapsed_time = time.strftime("%H:%M:%S",
                                 time.gmtime(time.time()-start_time))
    save_results(test_id, n_features, n_layers, n_epoch, n_batch,
                 regularization, rmse_all_counts, rmse_epsilon_all_counts,
                 pearson_all_counts, elapsed_time)
