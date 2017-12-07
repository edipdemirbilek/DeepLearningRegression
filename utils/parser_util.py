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
                        choices=['dl', 'rf', 'bg'],
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
    parser.add_argument('--n_features', type=int, choices=["1 - 125"],
                        default=None,
                        help="number of features")
    parser.add_argument('--count', type=int, default=3,
                        help="repeat 'count' times k-fold cross validation on \
                        the same model")
    return parser


def add_dl_parser_arguments(parser):
    """
    Builds argument parser.

    Arguments:
        parser - ArgumentParser initialized with common arguments

    Returns:
        parser -- ArgumentParser with dl arguments
    """
    parser.add_argument('--n_layer', type=int,
                        default=None, choices=["1 - 20"],
                        help="max number of hidden layers")
    parser.add_argument('--n_epoch', type=int,
                        default=None, choices=["1 - 2^14"],
                        help='max number of epochs')
    parser.add_argument('--n_batch', choices=["1-120"], default=120,
                        help='batch number')
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
    parser.add_argument('--n_trees', type=int,
                        default=None, choices=["1 - 200"],
                        help="number of trees in the forest")
    parser.add_argument('--max_features',
                        default=None, choices=['int', 'float', "auto", "sqrt",
                                               "log2", "None"],
                        help="number of features to consider when looking for \
                            the best split")
    parser.add_argument('--max_depth', default=None,
                        choices=['int', "None"],
                        help="maximum depth of the tree")
    # criterion
    # min_samples_split
    # min_samples_leaf
    # min_weight_fraction_leaf
    # max_leaf_nodes
    # min_impurity_split
    # min_impurity_decrease
    # bootstrap
    # oob_score
    # n_jobs
    # random_state
    # warm_start


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
