# -*- coding: utf-8 -*-
"""
@author: edip.demirbilek

PCA Util.

This module allow us to generate PCA based feature extraction.
"""
from numpy import asarray, concatenate
from sklearn.decomposition import PCA

import pandas as pd

# File Names
PCA_FEATURES_FILE_PREFIX = "./pca_features/pca_"


def prepare_data_for_pca(attributes, labels):
    """
    From attributes that also includes confidence intervals, and labels lists
    creates two arrays. One includes only attribues. Another that includes
    confidence intervals and labels.

    Arguments:
        attributes -- attributes with confidence intervals, list(160x127)
        labels -- labels, list(160,)

    Returns:
        attributes_as_array -- attributes without confidence intervals,
            array(160x125)
        ci_labels_as_array -- confidence intervals and labels, (160x3)
    """
    # list(160x127) -> array(160x127)
    attributes_as_array = asarray(attributes)
    # print(attributes_as_array.shape)

    # array(160x2), confidence intervals in last two columns
    ci_as_array = attributes_as_array[:, 125:]
    # print(ci_as_array.shape)

    # list(160,) -> array(160,)
    labels_as_array = asarray(labels)
    # array(160x1)
    labels_as_array = labels_as_array.reshape(labels_as_array.shape[0], 1)
    # print(labels_as_array.shape)

    ci_labels_as_array = concatenate((ci_as_array, labels_as_array), axis=1)
    # print(ci_labels_as_array.shape)

    # last two columns are confidence intervals (160x125)
    attributes_as_array = attributes_as_array[:, 0:125]
    # print(attributes_as_array.shape)
    return attributes_as_array, ci_labels_as_array


def save_pca_features(count, attributes_ci_labels_as_array):
    """
    Saves attributes, confidence interval values and labels to csv file.

    Arguments:
        count -- number of attributes, int
        attributes_ci_labels_as_array -- attributes, ci values and labels
            (160xcount+3)

    Returns:
        None
    """
    print("Saving "+str(count)+" pca extracted features")
    columns = []
    for step in range(1, count+1):
        columns.append("pca-"+str(step))

    columns.append("CIHigh")
    columns.append("CILow")
    columns.append("MOS")

    pca_features_file_name = PCA_FEATURES_FILE_PREFIX+str(count)+".csv"

    df = pd.DataFrame(attributes_ci_labels_as_array, columns=columns)
    df.to_csv(pca_features_file_name, index=False)


def generate_pca_features(attributes, labels, n_features):
    """
    Extracts features using PCA and saves to respective files in CSV format.

    Arguments:
        attributes -- attributes with confidence intervals, list(160x127)
        labels -- labels, list(160,)
        n_features -- int

    Returns:
        None
    """

    attributes_as_array, ci_labels_as_array = \
        prepare_data_for_pca(attributes, labels)

    for count in range(1, n_features+1):

        pca = PCA(n_components=count)

        # 160 x count
        attributes_new = pca.fit_transform(attributes_as_array)
        # print(attributes_new.shape)

        # count features, CIHigh, CILow, MOS (160 x count+2)
        attributes_ci_labels_as_array = \
            concatenate((attributes_new, ci_labels_as_array), axis=1)
        # print(attributes_ci_labels_as_array.shape)

        save_pca_features(count, attributes_ci_labels_as_array)