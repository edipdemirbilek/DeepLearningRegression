# -*- coding: utf-8 -*-
"""
@author: edip.demirbilek

Multimedia Audiovisual Quality Models.

This module includes multimedia audiovisual quality models based on
Deep Learning and Deceision Trees Ensamble MODELS developed using Bitstream
version of the  INRS Audiovisual Quality Dataset.

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
    check parser_util.py for available options
"""
import random

from numpy import power
from numpy.random import rand, uniform

from utils.dataset_util import load_dataset
from utils.dl_util import pack_regularization_object, \
    create_dl_model, run_dl_model, save_dl_header
from utils.rf_util import create_rf_model, run_rf_model, save_rf_header, \
    pack_rf_conf_object
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

        criterion = args.criterion if args.criterion else \
            random.choice(['mse', 'mae'])

        max_features = args.max_features if args.max_features else \
            random.choice(['int', 'float', 'auto', 'sqrt', 'log2', 'None'])
        if max_features == 'int':
            max_features = random.randint(1, n_features)
        elif max_features == 'float':
            max_features = uniform(0, 1.0)
        elif max_features == 'None':
            max_features = None

        max_depth = args.max_depth if args.max_depth else \
            random.choice(['int', 'None'])
        if max_depth == 'int':
            max_depth = random.randint(1, n_features)
        else:
            max_depth = None

        min_samples_split = args.min_samples_split if args.min_samples_split \
            else random.choice(['int', 'float'])
        if min_samples_split == 'int':
            min_samples_split = random.randint(2, 120)
        else:
            min_samples_split = uniform(0.0, 1.0)

        min_samples_leaf = args.min_samples_leaf if args.min_samples_leaf \
            else random.choice(['int', 'float'])
        if min_samples_leaf == 'int':
            min_samples_leaf = random.randint(1, 120)
        else:
            min_samples_leaf = uniform(0, 0.5)

        min_weight_fraction_leaf = args.min_weight_fraction_leaf if \
            args.min_weight_fraction_leaf else uniform(0, 0.5)

        max_leaf_nodes = args.max_leaf_nodes if args.max_leaf_nodes else \
            random.choice(['int', 'None'])
        if max_leaf_nodes == 'int':
            max_leaf_nodes = random.randint(1, n_features)
        else:
            max_leaf_nodes = None

        min_impurity_decrease = args.min_impurity_decrease if \
            args.min_impurity_decrease else uniform(0, 1.0)

        bootstrap = args.bootstrap if args.bootstrap else \
            random.choice([True, False])

        oob_score = args.oob_score if args.oob_score else \
            random.choice([True, False])
        if oob_score:
            bootstrap = True

        n_jobs = args.n_jobs if args.n_jobs else random.choice([-1, 1])

        warm_start = args.warm_start if args.warm_start else False

        rf_conf_object = \
            pack_rf_conf_object(n_trees, criterion, max_features, max_depth,
                                min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes,
                                min_impurity_decrease, bootstrap, oob_score,
                                n_jobs, warm_start, random_state=None)

        rf_model = create_rf_model(rf_conf_object)

        run_rf_model(attributes, labels, test_id, rf_model, args.count,
                     args.k, n_features, rf_conf_object, verbose=args.verbose)


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
