# -*- coding: utf-8 -*-
"""
@author: edip.demirbilek

Random Forest Util.

This module allow us to create/train/test and log evrything about Random
Forests based ensamble models.
"""
import time
import os

import pickle

from statistics import mean

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from utils.dataset_util import prepare_data, unpack_partitioned_data
from utils.common_util import accumulate_results_from_folds, compute_metrics, \
    compute_and_accumulate_results_from_counts


# File Names
RF_RESULTS_DETAILS_FILE_NAME = "rf_details.txt"
RF_RESULTS_SUMMARY_FILE_NAME = "rf_summary.csv"


def pack_rf_conf_object(n_trees, criterion, max_features, max_depth,
                        min_samples_split, min_samples_leaf,
                        min_weight_fraction_leaf, max_leaf_nodes,
                        min_impurity_decrease, bootstrap, oob_score, n_jobs,
                        warm_start, random_state):
    """
    Packs rf configuration object.
    """
    rf_conf_object = {}
    rf_conf_object["n_trees"] = n_trees
    rf_conf_object["criterion"] = criterion
    rf_conf_object["max_features"] = max_features
    rf_conf_object["max_depth"] = max_depth
    rf_conf_object["min_samples_split"] = min_samples_split
    rf_conf_object["min_samples_leaf"] = min_samples_leaf
    rf_conf_object["min_weight_fraction_leaf"] = min_weight_fraction_leaf
    rf_conf_object["max_leaf_nodes"] = max_leaf_nodes
    rf_conf_object["min_impurity_decrease"] = min_impurity_decrease
    rf_conf_object["bootstrap"] = bootstrap
    rf_conf_object["oob_score"] = oob_score
    rf_conf_object["n_jobs"] = n_jobs
    rf_conf_object["warm_start"] = warm_start
    rf_conf_object["random_state"] = random_state

    return rf_conf_object


def unpack_rf_conf_object(rf_conf_object):
    """
    Unpacks rf configuration object.
    """
    n_trees = rf_conf_object["n_trees"]
    criterion = rf_conf_object["criterion"]
    max_features = rf_conf_object["max_features"]
    max_depth = rf_conf_object["max_depth"]
    min_samples_split = rf_conf_object["min_samples_split"]
    min_samples_leaf = rf_conf_object["min_samples_leaf"]
    min_weight_fraction_leaf = rf_conf_object["min_weight_fraction_leaf"]
    max_leaf_nodes = rf_conf_object["max_leaf_nodes"]
    min_impurity_decrease = rf_conf_object["min_impurity_decrease"]
    bootstrap = rf_conf_object["bootstrap"]
    oob_score = rf_conf_object["oob_score"]
    n_jobs = rf_conf_object["n_jobs"]
    warm_start = rf_conf_object["warm_start"]
    random_state = rf_conf_object["random_state"]

    return n_trees, criterion, max_features, max_depth, min_samples_split, \
        min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, \
        min_impurity_decrease, bootstrap, oob_score, n_jobs, warm_start, \
        random_state


def log_rf_hyperparameters(test_id, n_features, rf_conf_object):
    """
    Logs model's hyperparameters to DL_RESULTS_DETAILS_FILE_NAME file and
    stdout.

    Arguments:
        test_id -- test id, string
        n_features -- number of features, int
        rf_conf_object -- random forest conf parameters

    Returns:
        None
    """
    n_trees, criterion, max_features, max_depth, min_samples_split, \
        min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, \
        min_impurity_decrease, bootstrap, oob_score, n_jobs, warm_start, \
        random_state = unpack_rf_conf_object(rf_conf_object)

    log_string \
        = "\nTest Id: {}, Num Features: {}, Num Trees : {}, Criterion : {}, \
        Max Features: {}, Max Depth : {}, Min Samples Split : {}, \
        Min Samples Leaf : {}, Min Weight Fraction Leaf : {}, \
        Max Leaf Nodes : {}, Min Impurity Decrease : {}, Bootstrap : {}, \
        Oob Score: {}, Num Jobs : {}, Warm Start : {}, Random State : {},"\
        .format(test_id, n_features, n_trees, criterion, max_features,
                max_depth, min_samples_split, min_samples_leaf,
                min_weight_fraction_leaf, max_leaf_nodes,
                min_impurity_decrease, bootstrap, oob_score, n_jobs,
                warm_start, random_state)
    print(log_string)

    with open(RF_RESULTS_DETAILS_FILE_NAME, "a") as file:
        file.write(log_string)


def save_rf_header():
    """
    Saves CSV file header to RF_RESULTS_SUMMARY_FILE_NAME file.

    Arguments:
        None

    Returns:
        None
    """
    if not os.path.exists(RF_RESULTS_SUMMARY_FILE_NAME):
        with open(RF_RESULTS_SUMMARY_FILE_NAME, "a") as f:
            f.write("\ntest_id, n_features, n_trees, max_depth, max_features, \
                    rmse, rmse_epsilon, pearson, elapsed_time\n")


def save_rf_results(test_id, n_features, n_trees, max_depth, max_features,
                    rmse, rmse_epsilon, pearson, elapsed_time):
    """
    Saves averaged results of running a deep learning model over k-fold cross
    validation and repeating 'count' times to DL_RESULTS_SUMMARY_FILE_NAME and
    DL_RESULTS_DETAILS_FILE_NAME file.

    Arguments:
        test_id -- test id, string
        n_features -- number of features, int
        rmse -- root mean square error, list
        rmse_epsilon -- epsilon root mean square error, list
        pearson -- pearson correlation coefficient, list
        elapsed_time -- elapsed time, string

    Returns:
        None
    """
    result_for_print \
        = "Test Id: {}, Num Features: {}, Num Trees: {}, Max Depth: {}, \
        Max Features: {}, RMSE: {}, Epsilon RMSE: {}, Pearson: {}, \
        Elapsed Time: {}"\
            .format(test_id, n_features, n_trees, max_depth, max_features,
                    mean(rmse), mean(rmse_epsilon), mean(pearson),
                    elapsed_time)
    print("\nOverall Results:\n" + result_for_print)

    result_string_for_csv = '\n{},{},{},{},{},{},{},{},{}'\
        .format(test_id, n_features, n_trees, max_depth, max_features,
                mean(rmse), mean(rmse_epsilon), mean(pearson), elapsed_time)

    with open(RF_RESULTS_SUMMARY_FILE_NAME, "a") as file:
        file.write(result_string_for_csv)

    with open(RF_RESULTS_DETAILS_FILE_NAME, "a") as file:
        file.write(result_string_for_csv)


def create_rf_model(rf_conf_object):
    """
    Creates a random forests model using settings provided.

    Arguments:
        n_trees --
        depth --
        p_features --

    Returns:
        rf_model -- deep learning model
    """
    n_trees, criterion, max_features, max_depth, min_samples_split, \
        min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, \
        min_impurity_decrease, bootstrap, oob_score, n_jobs, warm_start, \
        random_state = unpack_rf_conf_object(rf_conf_object)

    rf_model = \
        RandomForestRegressor(n_estimators=n_trees, max_features=max_features,
                              max_depth=max_depth,
                              min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf,
                              min_weight_fraction_leaf=min_weight_fraction_leaf,
                              max_leaf_nodes=max_leaf_nodes,
                              min_impurity_decrease=min_impurity_decrease,
                              bootstrap=bootstrap, oob_score=oob_score,
                              n_jobs=n_jobs, random_state=random_state,
                              warm_start=warm_start)

    return rf_model


def train_rf_model(rf_model, x_train, y_train):
    """
    Trains the deep learning model.

    Arguments:
        rf_model -- random forests model
        x_train -- training attributes(data) of type numpy.ndarray and shape
            (training_data_size, n_features)
        y_train -- training labels of type numpy.ndarray and shape
            (training_data_size, )

    Returns:
        history -- model training history of type keras.callbacks.History
    """
    history = rf_model.fit(x_train, y_train)
    return history


def run_rf_model(attributes, labels, test_id, rf_model, count, k, n_features,
                 rf_conf_object, verbose):

    start_time = time.time()
    n_trees, criterion, max_features, max_depth, min_samples_split, \
        min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, \
        min_impurity_decrease, bootstrap, oob_score, n_jobs, warm_start, \
        random_state = unpack_rf_conf_object(rf_conf_object)

    log_rf_hyperparameters(test_id, n_features, rf_conf_object)

    rmse_all_counts = []
    rmse_epsilon_all_counts = []
    pearson_all_counts = []

    if rf_model is None:
        rf_model = create_rf_model(n_trees, max_depth, max_features)

    # save the model init state for later use
    with open('filename.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

#    if verbose:
#        print("\nModel Weights:\n" + str(model_weights))

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

            # reload the model with init state
            with open('filename.pkl', 'rb') as f:
                rf_model = pickle.load(f)

            train_rf_model(rf_model, x_train, y_train)

            prediction = rf_model.predict(x_test)
            # reshape from (n, ) to (n, 1)
            prediction = prediction.reshape(prediction.shape[0], 1)

            y_test_all_folds, prediction_all_folds, \
                prediction_epsilon_all_folds, prediction_epsilon \
                = accumulate_results_from_folds(
                    y_test_all_folds, prediction_all_folds,
                    prediction_epsilon_all_folds, i_fold, y_test,
                    prediction, ci_high, ci_low)

            if verbose:
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
    save_rf_results(test_id, n_features, n_trees, max_depth, max_features,
                    rmse_all_counts, rmse_epsilon_all_counts,
                    pearson_all_counts, elapsed_time)