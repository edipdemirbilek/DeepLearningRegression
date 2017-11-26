# -*- coding: utf-8 -*-
"""
@author: edip.demirbilek

Multimedia Audiovisual Quality Models.

This module includes Deep Learning and Deceision Trees Based Ensamble Models
developed using the Parametric and Bitstream version of the  INRS Audiovisual
Quality Dataset.

Module reads model type (Deep Learning or Decision Trees based ensamble
methods) and is capable of conductiong the random hyperparameter search for
preselected parameter range as well as running specific models tested with high
accuracy.

Usage Examples:

        $ python DLRegressionMOSKFold.py
        $ python DLRegressionMOSKFold.py --model_type=random
        $ python DLRegressionMOSKFold.py --model_type=random --debug=True
        $ python DLRegressionMOSKFold.py --model_type=custom
        $ python DLRegressionMOSKFold.py --model_type=custom --debug=True

Attributes:
    module_level_variable1 (int):

Todo:
    * Decision Trees based Ensamble Methods
    * Complete Docstrings
    * Add custom models
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
