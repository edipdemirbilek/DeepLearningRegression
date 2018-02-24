# -*- coding: utf-8 -*-
"""
@author: edip.demirbilek
"""
from numpy import array, zeros, concatenate
from sklearn.metrics import mean_squared_error

import numpy as np

import scipy as sp


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
        prediction_normalized -- normalized prediction labels (n, ?)
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

    if r_value < 0.84 or is_nan(r_value):
        raise ValueError("Pearson result is less than 0.85 or NaN. " +
                         " Skipping this configuration.")

    return rmse, rmse_epsilon, r_value, p_value


def compute_and_accumulate_results_from_counts(
        y_test_all_folds, prediction_all_folds, prediction_epsilon_all_folds,
        rmse_all_counts, rmse_epsilon_all_counts, pearson_all_counts, verbose):
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


def is_nan(x):
    return (x is np.nan or x != x)