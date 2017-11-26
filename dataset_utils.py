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
from numpy import array, asarray

DATASET_FILE_NAME = "BitstreamDataset_ColumnsSorted.csv"


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
            # split on comma
            row = line.strip().split(",")
            if first_row:
                first_row = False
                continue
            attributes_tmp.append(row)

    random.shuffle(attributes_tmp)

    # Separate attributes and labels
    attributes = []
    labels = []
    for row in attributes_tmp:
        labels.append(float(row.pop()))
        row_length = len(row)
        # eliminate ID
        attributes_one_row = [float(row[i]) for i in range(0, row_length)]
        attributes.append(attributes_one_row)

    # number of rows and columns in x matrix
    nrows = len(attributes)
    ncols = len(attributes[1])
    print("#Rows: " + str(nrows) + " #Cols: " + str(ncols))

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

    # This is for 0 mean and 1 variance
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train = (x_train - x_mean)/x_std
    x_test = (x_test - x_mean)/x_std

    # This is for normalization
    y_train = y_train/5
    y_test = y_test/5
    ci_low = array(ci_low)/5
    ci_high = array(ci_high)/5

    x_train = array(x_train)
    y_train = array(y_train)
    x_test = array(x_test)
    y_test = array(y_test)

    return x_train, y_train, x_test, y_test, ci_low, ci_high

# def save_resultsHeader():
#    with open(RESULTS_SUMMARY_FILE_NAME, "a") as f:
#        f.write("\ntest_id, num_features, n_layers, n_epoch, n_batch_size, \
# rmse, rmse_epsilon, pearson, elapsed_time, dropout, l2\n")
