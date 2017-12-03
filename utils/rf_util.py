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


def create_rf_model(n_trees, max_depth, max_features):
    """
    Creates a random forests model using settings provided.

    Arguments:
        n_trees --
        depth --
        p_features --

    Returns:
        rf_model -- deep learning model
    """
    rf_model = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth,
                                     max_features=max_features,
                                     random_state=None)

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
                 n_trees, max_depth, max_features, verbose):

    start_time = time.time()

#    log_hyperparameters(
#        test_id, n_features, n_layers, n_epoch, n_batch, regularization)

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
