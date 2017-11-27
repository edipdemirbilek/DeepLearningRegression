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

        $ python DLRegressionMOSKFold.py
            Creates and runs Deep Learning Custom Model 1.

        $ python DLRegressionMOSKFold.py --model_type=random
            Creates n Deep Learning models by randomly selecting
            hyperparameters over a large range.

        $ python DLRegressionMOSKFold.py --model_type=random --debug=True
            Creates n Deep Learning models by randomly selecting
            hyperparameters over a large range. Displays debug messages.

        $ python DLRegressionMOSKFold.py --model_type=custom
            Creates and runs Deep Learning Custom Model 1.

        $ python DLRegressionMOSKFold.py --model_type=custom --debug=True
            Creates and runs Deep Learning Custom Model 1. Displays debug
            messages.

Attributes:
    Common:
        debug -- debug, boolean, default False
        model_type -- string, dl/rf/bg, default dl
        random_search -- boolean, default False
        model_id -- string, default 1
        num_models -- int, default 500
        k -- int, default 4
        n_features -- int, default 127
        count -- int, default 3

    Deep Learning:
        n_layer_max -- int, default 20
        n_epoch_max -- int, default 2**14
        n_batch_size -- int, default 120
        dropout -- boolean, default random
        k_l2 -- boolean, default random
        k_l1 -- boolean, default random
        a_l2 -- boolean, default random
        a_l1 -- boolean, default random

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
from deep_learning.dl_utils import create_model, run_model
from deep_learning.dl_models import create_model_1

# for k-Fold cross validation
K = 4
NUM_FEATURES = 127


def build_parser():
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    parser = ArgumentParser()
    parser.add_argument('--debug', dest='debug',
                        help='debug level',
                        metavar='DEBUG', required=False)
    parser.add_argument('--model_type', dest='model_type',
                        help='model type(random, custom)',
                        metavar='MODEL TYPE', required=False)
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
        # For Random Search Hyperparameter exploration
        for i in range(1, 500):
            # n_layers = int(power(2,4*np.random.rand()))
            n_layers = random.randint(1, 20)
            n_epoch = int(power(2, 14 * uniform(0.642, 1.0)))
            n_batch_size = 120
            test_id = str(i)+str(rand())
            regularization = pack_regularization_object(
                dropout=random.choice([True, False]),
                k_l2=False, k_l1=True, a_l2=False, a_l1=False)
            dl_model = create_model(n_layers, NUM_FEATURES, regularization)
            run_model(attributes, labels, test_id, dl_model, count, K,
                      NUM_FEATURES, n_layers, n_epoch, n_batch_size,
                      regularization, debug=options.debug)
    else:
        count = 1
        test_id = str("custom") + str(rand())
        dl_model, n_layers, n_epoch, n_batch_size, regularization \
            = create_model_1(NUM_FEATURES)
        run_model(
            attributes, labels, test_id, dl_model, count, K, NUM_FEATURES,
            n_layers, n_epoch, n_batch_size, regularization,
            debug=options.debug)

if __name__ == '__main__':
    main()
