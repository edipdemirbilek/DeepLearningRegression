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
        n_features -- Number of features, int, default: 125, optional
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

from numpy import power
from numpy.random import rand, uniform

from utils.dataset_util import load_dataset
from utils.dl_util import pack_regularization_object, \
    create_dl_model, run_dl_model, save_dl_header
from utils.rf_util import create_rf_model, run_rf_model, save_rf_header
from utils.parser_util import build_parser
from models.dl_models import dl_model_1, dl_model_2, dl_model_3, \
    dl_model_4, dl_model_5, dl_model_6, dl_model_7, dl_model_8, dl_model_9, \
    dl_model_10


def process_dl_random_model(args, attributes, labels):
    """
    Deep Learning Random Search Hyperparameter exploration parses input
    arguments and creates and runs random models.

    Arguments:
        args -- a number of arguments. See top level dostrings for
            detailed options available.
        attributes -- training/test data
        labels -- training/test labels

    Returns:
        None

    Raises:
        None
    """
    for i in range(1, args.n_models):
        test_id = str(i)+str(rand())

        n_features = args.n_features if args.n_features else \
            int(power(2, 7 * uniform(0, 0.995112040666012)))

        n_layer = args.n_layer if args.n_layer else \
            int(power(2, 5 * uniform(0, 1.0)))

        n_epoch = args.n_epoch if args.n_epoch else \
            int(power(2, 14 * uniform(0.617418299269623, 1.0)))

        dropout = args.dropout if args.dropout else \
            random.choice([True, False])

        k_l2 = args.k_l2 if args.k_l2 else random.choice([True, False])
        k_l1 = args.k_l1 if args.k_l1 else random.choice([True, False])
        a_l2 = args.a_l2 if args.a_l2 else random.choice([True, False])
        a_l1 = args.a_l1 if args.a_l1 else random.choice([True, False])

        regularization = pack_regularization_object(
            dropout=dropout, k_l2=k_l2, k_l1=k_l1,
            a_l2=a_l2, a_l1=a_l1)

        dl_model = create_dl_model(n_layer, n_features,
                                   regularization)

        run_dl_model(attributes, labels, test_id, dl_model, args.count,
                     args.k, n_features, n_layer, n_epoch,
                     args.n_batch, regularization,
                     verbose=args.verbose)


def process_dl_custom_model(args, attributes, labels):
    """
    Parses input arguments and creates and runs custom deep learning based
    models.

    Arguments:
        args -- a number of arguments. See top level dostrings for
            detailed options available.
        attributes -- training/test data
        labels -- training/test labels

    Returns:
        None

    Raises:
        None
    """
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

    test_id = "dl_model_"+str(args.model_id)
    save_dl_header()

    n_features = args.n_features if args.n_features else 125

    dl_model, n_layers, n_epoch, n_batch_size, regularization \
        = custom_dl_models[args.model_id](n_features)
    run_dl_model(
        attributes, labels, test_id, dl_model, args.count, args.k,
        n_features, n_layers, n_epoch, n_batch_size,
        regularization, verbose=args.verbose)


def process_rf_random_model(args, attributes, labels):
    """
    Random Forests Random Search Hyperparameter exploration parses input
    arguments and creates and runs random models.

    Arguments:
        args -- a number of arguments. See top level dostrings for
            detailed options available.
        attributes -- training/test data
        labels -- training/test labels

    Returns:
        None

    Raises:
        None
    """
    for i in range(1, args.n_models):
        test_id = str(i)+str(rand())

        n_features = args.n_features if args.n_features else \
            random.randint(1, 125)

        n_trees = args.n_trees if args.n_trees else \
            int(power(2, 11 * uniform(0, 1.0)))

        max_depth = args.max_depth if args.max_depth else None
        max_features = args.max_features if args.max_features else \
            "auto"

        rf_model = create_rf_model(n_trees, max_depth, max_features)

        run_rf_model(attributes, labels, test_id, rf_model, args.count,
                     args.k, n_features, n_trees, max_depth,
                     max_features, verbose=args.verbose)


def main():
    """
    Main function parses input arguments and creates and runs random/custom
    models based on deep learning and random forests.

    Arguments:
        arguments -- a number of arguments. See top level dostrings for
        detailed options available.

    Returns:
        None

    Raises:
        None
    """
    parser = build_parser()
    args = parser.parse_args()

    attributes, labels = load_dataset()

    if args.m_type == "dl":
        save_dl_header()

        if args.random:
            process_dl_random_model(args, attributes, labels)
        else:
            process_dl_custom_model(args, attributes, labels)

    elif args.m_type == "rf":
        save_rf_header()

        if args.random:
            process_rf_random_model(args, attributes, labels)


if __name__ == '__main__':
    main()
