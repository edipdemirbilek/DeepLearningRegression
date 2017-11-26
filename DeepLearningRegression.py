# -*- coding: utf-8 -*-
"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python DLRegressionMOSKFold.py --model_type=random
        $ python DLRegressionMOSKFold.py --model_type=random --debug=True
        $ python DLRegressionMOSKFold.py --model_type=custom
        $ python DLRegressionMOSKFold.py --model_type=custom --debug=True
        

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import random
import sys
import time

from statistics import mean

from numpy import array, asarray, power, zeros, concatenate
from numpy.random import rand, uniform
import scipy as sp

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.optimizers import Adadelta
from keras import regularizers

from argparse import ArgumentParser

# File Names
DATASET_FILE_NAME = "BitstreamDataset_ColumnsSorted.csv"
RESULTS_DETAILS_FILE_NAME = "Details_new.txt"
RESULTS_SUMMARY_FILE_NAME = "Results_new.csv"

# for k-Fold cross validation
K = 4
NUM_FEATURES = 127

def load_dataset():
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    # arrange data into list for labels and list of lists for attributes
    attributes_tmp = []
    first_row = True
    
    with open(DATASET_FILE_NAME) as file:
        for line in file:
            #split on comma
            row = line.strip().split(",")
            if first_row:
                first_row = False
                continue
            attributes_tmp.append(row)
            
    random.shuffle(attributes_tmp)
    
    #Separate attributes and labels
    attributes = []
    labels = []
    for row in attributes_tmp:
        labels.append(float(row.pop()))
        row_length = len(row)
        #eliminate ID
        attributes_one_row = [float(row[i]) for i in range(0, row_length)]
        attributes.append(attributes_one_row)
        
    #number of rows and columns in x matrix
    nrows = len(attributes)
    ncols = len(attributes[1])
    print("#Rows: "+str(nrows) +" #Cols: "+str(ncols))
    
    return attributes, labels

def prepare_data(attributes, train, test, labels, n_features):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    x_train_all_attributes, x_test_all_attributes, y_train, y_test \
        = [attributes[i] for i in train], \
          [attributes[i] for i in test], \
          [labels[i] for i in train], \
          [labels[i] for i in test]
    
    # x_train[0][1:n_features+1]
    x_train = [] 
    x_test = []
    ci_high = []
    ci_low = []
    
    for x_train_row in x_train_all_attributes:
        x_train.append(x_train_row[0:n_features])
        
    for x_test_row in x_test_all_attributes:
        x_test.append(x_test_row[0:n_features])
        ci_high.append(x_test_row[-2:-1])
        ci_low.append(x_test_row[-1:])
        
    x_train = asarray(x_train)
    x_test = asarray(x_test)
    y_train = asarray(y_train)
    y_test = asarray(y_test)
    
    #This is for 0 mean and 1 variance
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train = (x_train - x_mean)/x_std
    x_test = (x_test - x_mean)/x_std
    
    #This is for normalization
    y_train = y_train/5
    y_test = y_test/5
    ci_low = array(ci_low)/5
    ci_high = array(ci_high)/5
    
    x_train = array(x_train)
    y_train = array(y_train)   
    x_test = array(x_test)
    y_test = array(y_test)
    
    return x_train, y_train, x_test, y_test, ci_low, ci_high

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
    dropout, k_l2, k_l1, a_l2, a_l1 = unpack_regularization_object(regularization)
    with open(RESULTS_DETAILS_FILE_NAME, "a") as file:
        if dropout:
            file.write("\n    dropout: "+ str(dropout))
            file.write("\n    rate: "+ str(rate))
            print("    dropout: "+ str(dropout))
            print("    rate: "+ str(rate))

        file.write("\n    layer: "+ str(layer))
        file.write("\n    n_hidden: "+ str(n_hidden))

        if k_l2:
            file.write("\n    k_l2: "+ str(k_l2))
        if k_l1:
            file.write("\n    k_l1: "+ str(k_l1))
        if k_l2 or k_l1: 
            file.write("\n    k_v: "+ str(k_v))
        if a_l2:
            file.write("\n    a_l2: "+ str(a_l2))
        if a_l1:
            file.write("\n    a_l1: "+ str(a_l1))
        if a_l2 or a_l1:
            file.write("\n    a_v: "+ str(a_v))
            
        print("    layer: "+ str(layer))
        print("    n_hidden: "+ str(n_hidden))
        
        if k_l2:
            print("    k_l2: "+ str(k_l2))
        if k_l1:
            print("    k_l1: "+ str(k_l1))
        if k_l2 or k_l1: 
            print("    k_v: "+ str(k_v))
        if a_l2:
            print("    a_l2: "+ str(a_l2))
        if a_l1:
            print("    a_l1: "+ str(a_l1))
        if a_l2 or a_l1:
            print("    a_v: "+ str(a_v))


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
    k_regularizer, a_regularizer, k_v, a_v = initialize_layer_parameters(regularization)
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

        regularization = pack_regularization_object(dropout, k_l2, k_l1, a_l2, a_l1)
        k_regularizer, a_regularizer, k_v, a_v = initialize_layer_parameters(regularization)

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

def train_model(dl_model, x_train, y_train, n_batch_size, n_epoch, x_test, y_test):
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
        
def log_hyperparameters(test_id, n_features, n_layers, n_epoch, n_batch_size, regularization):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    dropout, k_l2, k_l1, a_l2, a_l1 = unpack_regularization_object(regularization)
    log_string \
        = "\nTest Id: {}, Num Features: {}, Num Layers: {}, Num Epochs: {}, Num Batch Size: {}, Dropout: {}, k_l2: {}, k_l1: {}, a_l2: {}, a_l1: {}"\
        .format(test_id, n_features, n_layers, n_epoch, n_batch_size, dropout, k_l2, k_l1, a_l2, a_l1)
    print(log_string)
        
    with open(RESULTS_DETAILS_FILE_NAME, "a") as file:
        file.write(log_string)

#def save_resultsHeader():
#    with open(RESULTS_SUMMARY_FILE_NAME, "a") as f:
#        f.write("\ntest_id, num_features, n_layers, n_epoch, n_batch_size, \
# rmse, rmse_epsilon, pearson, elapsed_time, dropout, l2\n")
        
def save_results(test_id, n_features, n_layers, n_epoch, n_batch_size, regularization, 
                 rmse_per_count, rmse_epsilon_per_count, pearson_per_count, elapsed_time):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    dropout, k_l2, k_l1, a_l2, a_l1 = unpack_regularization_object(regularization)
    result_for_print \
        = "Test Id: {}, Num Features: {}, Num Layers: {}, Num Epochs: {}, Num Batch Size: {}, Dropout: {}, k_l2: {}, k_l1: {}, a_l2: {}, a_l1: {}, RMSE: {}, Epsilon RMSE: {}, Pearson: {}, Elapsed Time: {}"\
            .format(test_id, n_features, n_layers, n_epoch, n_batch_size, dropout, 
                    k_l2, k_l1, a_l2, a_l1, mean(rmse_per_count), 
                    mean(rmse_epsilon_per_count), mean(pearson_per_count), 
                    elapsed_time)
    print("\nOverall Results:\n" + result_for_print)
    
    result_string_for_csv = '\n{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
        test_id, n_features, n_layers, n_epoch, n_batch_size, 
        mean(rmse_per_count), mean(rmse_epsilon_per_count), 
        mean(pearson_per_count), elapsed_time, dropout, 
        k_l2, k_l1, a_l2, a_l1)
            
    with open(RESULTS_SUMMARY_FILE_NAME, "a") as file:
        file.write(result_string_for_csv)
        
    with open(RESULTS_DETAILS_FILE_NAME, "a") as file:
        file.write(result_string_for_csv)

    
def accumulate_results_from_folds(y_test_for_all_folds, prediction_folds, 
                                  prediction_epsilon_folds, i_fold, 
                                  prediction_from_fold, ci_low, ci_high, y_test):
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
        
    return y_test_for_all_folds, prediction_folds, \
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
        print("            y_test Normalized: "
              + ', '.join(["%.2f" % e for e in y_test_normalized]))
        print("        Prediction Normalized: "
              +', '.join(["%.2f" % e for e in prediction_normalized]))
        print("Epsilon Prediction Normalized: "
              +', '.join(["%.2f" % e for e in prediction_epsilon_normalized]))
    
    if debug:
        print("                       y_test: "
              +', '.join(["%.2f" % e for e in y_test]))
        print("                   Prediction: "
              +', '.join(["%.2f" % e for e in prediction]))
        print("           Epsilon Prediction: "
              +', '.join(["%.2f" % e for e in prediction_epsilon]))
    
    #mse=mean_squared_error(y_test, prediction_from_fold)
    #rmse https://www.kaggle.com/wiki/RootMeanSquaredError
    
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
        print("\nModel Weights:\n" +str(model_weights))
    
    for count in range(1, count + 1):    
        print("\nCount: " + str(count)+ " Time: " + time.ctime())
        
        y_test_for_all_folds = []
        prediction_folds = []
        prediction_epsilon_folds = []
        i_fold = 0
        
        k_fold = KFold(n_splits=k)
        for train_index, test_index in k_fold.split(attributes):
            x_train, y_train, x_test, y_test, ci_low, ci_high = prepare_data(
                attributes, train_index, test_index, labels, n_features)  
            dl_model.set_weights(model_weights)
            train_model(
                dl_model, x_train, y_train, n_batch_size, n_epoch, x_test, y_test)
            prediction_from_fold = dl_model.predict(x_test)
            
            y_test_for_all_folds, prediction_folds, \
            prediction_epsilon_folds, prediction_epsilon \
                = accumulate_results_from_folds(
                    y_test_for_all_folds, prediction_folds, 
                    prediction_epsilon_folds, i_fold, 
                    prediction_from_fold, ci_low, ci_high, y_test)
            
            if debug:
                print("\nMetrics for fold: " + str(i_fold + 1))
                compute_metrics(y_test, prediction_from_fold, prediction_epsilon, debug)
                
            i_fold += 1
        
        if debug:
            print("\nMetrics for count: " + str(count))
        rmse_per_count, rmse_epsilon_per_count, pearson_per_count \
            = compute_results(y_test_for_all_folds, prediction_folds, 
                              prediction_epsilon_folds, rmse_per_count, 
                              rmse_epsilon_per_count, pearson_per_count, debug)

    elapsed_time = time.strftime("%H:%M:%S", \
                                 time.gmtime(time.time()-start_time))
    save_results(
        test_id, n_features, n_layers, n_epoch, n_batch_size, regularization, 
        rmse_per_count, rmse_epsilon_per_count, pearson_per_count, 
        elapsed_time)

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
    regularization = pack_regularization_object(dropout, k_l2, k_l1, a_l2, a_l1)
    
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

def build_parser():
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    parser = ArgumentParser()
    parser.add_argument('--debug',
            dest = 'debug', help = 'debug level',
            metavar = 'DEBUG', required = False)
    parser.add_argument('--model_type',
            dest = 'model_type', help = 'model type(random, custom)',
            metavar = 'MODEL TYPE', required = True)

    return parser

def main():
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    parser = build_parser()
    options = parser.parse_args()
    
    attributes, labels = load_dataset()
    
    model_type_is_random = options.model_type == "random"
    
    if model_type_is_random:
        # save_resultsHeader()
        # running the same test count times
        count = 3
        ## For Random Search Hyperparameter exploration    
        for i in range(1, 500):            
            # n_layers = int(power(2,4*np.random.rand()))
            n_layers = random.randint(1, 20)
            n_epoch = int(power(2, 14 * uniform(0.642, 1.0)))
            n_batch_size = 120
            test_id = str(i)+str(rand())
            regularization = pack_regularization_object(
                dropout=random.choice([True, False]), 
                k_l2=False, k_l1=True, a_l2=False, a_l1=False)
            run_model(attributes, labels, test_id, None, count, K, NUM_FEATURES, 
                      n_layers, n_epoch, n_batch_size, regularization, debug=options.debug) 
    else:
        count = 1
        test_id = str("custom") + str(rand())    
        dl_model, n_layers, n_epoch, n_batch_size, regularization \
            = create_model_1(NUM_FEATURES)    
        run_model(
            attributes, labels, test_id, dl_model, count, K, NUM_FEATURES, 
            n_layers, n_epoch, n_batch_size, regularization, debug=options.debug)
    
if __name__ == '__main__':
    main()    
 