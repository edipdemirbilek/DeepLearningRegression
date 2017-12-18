#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: edip.demirbilek

ArgumentParser builder. See main nodule (regression_models.py) for overview of
options available.
"""
from argparse import ArgumentParser


def build_common_parser():
    """
    Builds argument parser.

    Arguments:
        None.

    Returns:
        parser -- Argument parser containing common arguments
    """
    parser = ArgumentParser()

    # common parameters
    parser.add_argument('-v', '--verbose',
                        help='increase output verbosity',
                        action="store_true")
    parser.add_argument('--m_type', required=True,
                        choices=['dl', 'rf', 'pca', 'fast_ica',
                                 'incremental_pca', 'kernel_pca'],
                        help='model type: deep learning, random forests or \
                        bagging')
    parser.add_argument('--random', action="store_true", help="random search \
                        over hyperparameters")
    parser.add_argument('--model_id', type=int, default=1,
                        help="custom model id")
    parser.add_argument('--n_models', type=int, default=500,
                        help="max number of models to create during random \
                        search over hyperparameters")
    parser.add_argument('--k', type=int, default=4,
                        help="cross validation k-fold value")
    parser.add_argument('--n_features', type=int, choices=range(1, 126),
                        default=None,
                        help="number of features")
    parser.add_argument('--count', type=int, default=3,
                        help="repeat 'count' times k-fold cross validation on \
                        the same model")
    parser.add_argument('--f_type', choices=['sorted', 'pca', 'fast_ica',
                                             'incremental_pca', 'kernel_pca'],
                        default=None, help='features to use')
    return parser


def add_dl_parser_arguments(parser):
    """
    Builds argument parser.

    Arguments:
        parser - ArgumentParser initialized with common arguments

    Returns:
        parser -- ArgumentParser with dl arguments
    """
    # model hyper parameters
    parser.add_argument('--n_layers', type=int,
                        default=None, choices=["1 - 20"],
                        help="max number of hidden layers")
    parser.add_argument('--n_epoch', type=int,
                        default=None, choices=["1 - 2^14"],
                        help='max number of epochs')
    parser.add_argument('--n_batch', choices=["1-120"], default=120,
                        help='batch number')
    parser.add_argument('--loss', default=None,
                        choices=['mean_squared_error', 'mean_absolute_error',
                                 'mean_absolute_percentage_error',
                                 'mean_squared_logarithmic_error',
                                 'squared_hinge', 'hinge', 'categorical_hinge',
                                 'logcosh', 'sparse_categorical_crossentropy',
                                 'binary_crossentropy',
                                 'kullback_leibler_divergence', 'poisson',
                                 'cosine_proximity'],
                        help='loss function')
    parser.add_argument('--optimizer', default=None,
                        choices=['SGD', 'RMSprop', 'Adagrad', 'Adadelta',
                                 'Adam', 'Adamax', 'Nadam'],
                        help='optimizer')

    # layer hyper parameters
    parser.add_argument('--h_activation', default=None,
                        choices=['softmax', 'elu', 'selu', 'softplus',
                                 'softsign', 'relu', 'tanh', 'sigmoid',
                                 'hard_sigmoid', 'linear'],
                        help='hidden layer activation function')
    parser.add_argument('--o_activation', default=None,
                        choices=['softmax', 'elu', 'selu', 'softplus',
                                 'softsign', 'relu', 'tanh', 'sigmoid',
                                 'hard_sigmoid', 'linear'],
                        help='output layer activation function')
    parser.add_argument('--k_initializer', default=None,
                        choices=['zeros', 'ones', 'random_normal',
                                 'random_uniform', 'truncated_normal',
                                 'orthogonal', 'lecun_uniform',
                                 'glorot_normal', 'glorot_uniform',
                                 'he_normal', 'lecun_normal', 'he_uniform'],
                        help='kernel initializer')

    # regularization settings
    parser.add_argument('--dropout', action="store_true",
                        default=None,
                        help="apply dropout")
    parser.add_argument('--k_l2', action="store_true",
                        default=None,
                        help='kernel L2 regularization')
    parser.add_argument('--k_l1', action="store_true",
                        default=None,
                        help='kernel L1 regularization')
    parser.add_argument('--a_l2', action="store_true",
                        default=None,
                        help='activation L2 regularization')
    parser.add_argument('--a_l1', action="store_true",
                        default=None,
                        help='activation L1 regularization')


def add_rf_parser_arguments(parser):
    """
    Builds argument parser.

    Arguments:
        parser - ArgumentParser initialized with common arguments

    Returns:
        parser -- ArgumentParser with rf arguments
    """
    parser.add_argument('--n_trees', type=int, default=None,
                        help="number of trees in the forest")
    parser.add_argument('--criterion', default=None, choices=['mse', 'mae'],
                        help="function to measure the quality of a split")
    parser.add_argument('--max_features',
                        default=None, choices=['int', 'float', "auto", "sqrt",
                                               "log2", "None"],
                        help="number of features to consider when looking for \
                            the best split")
    parser.add_argument('--max_depth', default=None, choices=['int', 'None'],
                        help="maximum depth of the tree")
    parser.add_argument('--min_samples_split', choices=['int', 'float'],
                        default=None,
                        help="minimum number of samples required to split an \
                        internal node")
    parser.add_argument('--min_samples_leaf', choices=['int', 'float'],
                        default=None,
                        help="minimum number of samples required to be at a \
                        leaf node")
    parser.add_argument('--min_weight_fraction_leaf', type=float,
                        default=None, help="minimum weighted fraction")
    parser.add_argument('--max_leaf_nodes', choices=['int', 'None'],
                        default=None,
                        help="grow trees with max_leaf_nodes in best-first \
                        fashion")
    parser.add_argument('--min_impurity_decrease', type=float,
                        default=None, help="A node will be split if this split \
                        induces a decrease of the impurity greater than or \
                        equal to this value")
    parser.add_argument('--bootstrap', type=bool,
                        default=None, help="whether bootstrap samples are used \
                        when building trees")
    parser.add_argument('--oob_score', type=bool, default=None,
                        help='whether to use out-of-bag samples to estimate \
                        the R^2 on unseen data')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='the number of jobs to run in parallel for both \
                        fit and predict. If -1, then the number of jobs is set\
                         to the number of cores')
    parser.add_argument('--warm_start', type=bool, default=None,
                        help='when set to True, reuse the solution of the \
                        previous call to fit and add more estimators to the \
                        ensemble, otherwise, just fit a whole new forest')


def build_parser():
    """
    Builds argument parser.

    Arguments:
        None.

    Returns:
        parser -- Argument parser. See top level docstrings for detailed
            options available.
    """
    parser = build_common_parser()
    add_dl_parser_arguments(parser)
    add_rf_parser_arguments(parser)

    return parser