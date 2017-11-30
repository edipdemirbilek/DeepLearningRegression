# -*- coding: utf-8 -*-
"""
@author: edip.demirbilek

Multimedia Audiovisual Quality Models.

This module includes multimedia audiovisual quality models based on
Deep Learning and Deceision Trees Ensamble MODELS developed using the
Parametric and Bitstream version of the  INRS Audiovisual Quality Dataset.

Module reads model type (Deep Learning or Decision Trees based ensamble
methods) and is capable of conductiong the random hyperparameter search for
a large preselected parameter range as well as running specific models tested
with high accuracy.

Usage Examples:

        $ python regression_models.py
            This would not run as --m_type parameter is mandatory

        $ python regression_models.py --m_type=dl
            This would run dl custom model with id=1

        $ python regression_models.py --m_type=dl --random
            This would create n=500 deep learning based models by randomly
            selecting hyperparameters using the default values.

        $ python regression_models.py --h
            This would print help. Use help or docstrings for a complete list
            of parameters and operations available.

Attributes:
    Common:
        verbose -- Verbose, default: disabled, optional
        m_type -- Model type, string, options: 'dl', 'rf' or 'bg', required
        random --  Random hyperparameter search, default: disabled, optional
        model_id -- Custom model id, int, default: 1, optional
        n_models -- Max number of models, int, default: 500, optional
        k -- Cross validation k-fold value, int, default: 4, optional
        n_features -- Number of features, int, default: 127, optional
        count -- Repeat k-fold cross validation # times, int, default: 3,
            optional

    Deep Learning: , only for random search
        n_layer -- Max number of hidden layers, int, default: random(1, 20),
            optional
        n_epoch -- Max number of epochs, int, default: random(512,2^14),
            optional
        n_batch -- Batch size, int, default: 120, optional
        dropout -- Dropout, default: random(True, False), optional
        k_l2 -- Kernel L2 regularization, default: False, optional
        k_l1 -- Kernel L1 regularization, default: True, optional
        a_l2 -- Activation L2 regularization, default: False, optional
        a_l1 -- Activation L1 regularization, default: False, optional

    Random Forests:

Todo:
    * Decision Trees based Ensamble Methods
    * Complete Docstrings
"""

import random

from argparse import ArgumentParser

from numpy import power
from numpy.random import rand, uniform

from dataset_utils import load_dataset
from deep_learning.dl_utils import pack_regularization_object
from deep_learning.dl_utils import create_model, run_model, save_header
from deep_learning.dl_models import dl_model_1, dl_model_2, dl_model_3, \
    dl_model_4, dl_model_5, dl_model_6, dl_model_7, dl_model_8, dl_model_9, \
    dl_model_10


def build_parser():
    """
    Builds argument parser.

    Arguments:
        None.

    Returns:
        parser -- Argument parser. See top level docstrings for detailed
            options available.
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
    parser.add_argument('--n_models', type=int, default=500, help="max number of \
                        models to create during random search over \
                        hyperparameters")
    parser.add_argument('--k', type=int, default=4,
                        help="cross validation k-fold value")
    parser.add_argument('--n_features', type=int, choices=["1 - 127"],
                        default=127, help="number of features")
    parser.add_argument('--count', type=int, default=3,
                        help="repeat 'count' times k-fold cross validation on \
                        the same model")

    # Deep Learning Parameters
    parser.add_argument('--n_layer', type=int, default=random.randint(1, 20),
                        choices=["1 - 20"],
                        help="max number of hidden layers")
    parser.add_argument('--n_epoch', type=int,
                        default=int(power(2, 14 * uniform(0.642, 1.0))),
                        choices=["1 - 2^14"], help='max number of \
                        epochs')
    parser.add_argument('--n_batch', choices=["1-127"], default=120,
                        help='batch number')
    parser.add_argument('--dropout', action="store_true",
                        default=random.choice([True, False]),
                        help="apply dropout")
    parser.add_argument('--k_l2', action="store_true",
                        default=False,
                        help='kernel L2 regularization')
    parser.add_argument('--k_l1', action="store_true",
                        default=True,
                        help='kernel L1 regularization')
    parser.add_argument('--a_l2', action="store_true",
                        default=False,
                        help='activation L2 regularization')
    parser.add_argument('--a_l1', action="store_true",
                        default=False,
                        help='activation L1 regularization')
    return parser


def main():
    """
    Main function parses input arguments and creates and runs random/custom
    models based on deep learning.

    Arguments:
        arguments -- a number of arguments. See top level dostrings for
        detailed options available.

    Returns:
        None

    Raises:
        Exception -- when a not implemented algorithm (dl or bg) is selected.
    """
    parser = build_parser()
    args = parser.parse_args()

    attributes, labels = load_dataset()

    if args.m_type == "dl":
        if args.random:
            save_header()
            # For Random Search Hyperparameter exploration
            for i in range(1, args.n_models):
                test_id = str(i)+str(rand())

                regularization = pack_regularization_object(
                    dropout=args.dropout, k_l2=args.k_l2, k_l1=args.k_l1,
                    a_l2=args.a_l2, a_l1=args.a_l1)

                dl_model = create_model(args.n_layer, args.n_features,
                                        regularization)

                run_model(attributes, labels, test_id, dl_model, args.count,
                          args.k, args.n_features, args.n_layer, args.n_epoch,
                          args.n_batch, regularization, verbose=args.verbose)
        else:

            custom_dl_models = {1: dl_model_1,
                                2: dl_model_2,
                                3: dl_model_3,
                                4: dl_model_4,
                                5: dl_model_5,
                                6: dl_model_6,
                                7: dl_model_7,
                                8: dl_model_8,
                                9: dl_model_9,
                                10: dl_model_10}

            test_id = str("custom") + str(rand())

            dl_model, n_layers, n_epoch, n_batch_size, regularization \
                = custom_dl_models[args.model_id](args.n_features)
            run_model(
                attributes, labels, test_id, dl_model, args.count, args.k,
                args.n_features, n_layers, n_epoch, n_batch_size,
                regularization, verbose=args.verbose)
    else:
        raise Exception('Not Implemented')

if __name__ == '__main__':
    main()
