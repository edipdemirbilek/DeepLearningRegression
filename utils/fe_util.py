# -*- coding: utf-8 -*-
"""
@author: edip.demirbilek

PCA Util.

This module allow us to generate PCA based feature extraction.
"""
from numpy import asarray, concatenate
import pandas as pd

from sklearn.decomposition import PCA, FastICA, IncrementalPCA, KernelPCA

# File Names
PCA_FEATURES_FILE_PREFIX = "./dataset/pca_extracted/pca_"
FAST_ICA_FEATURES_FILE_PREFIX = "./dataset/fast_ica_extracted/fast_ica_"
INCREMENTAL_PCA_FEATURES_FILE_PREFIX = \
    "./dataset/incremental_pca_extracted/incremental_pca_"
KERNEL_PCA_FEATURES_FILE_PREFIX = "./dataset/kernel_pca_extracted/kernel_pca_"


def prepare_data_for_feature_extraction(attributes, labels):
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


def save_extracted_features(filename_prefix, count,
                            attributes_ci_labels_as_array):
    """
    Saves attributes, confidence interval values and labels to csv file.

    Arguments:
        count -- number of attributes, int
        attributes_ci_labels_as_array -- attributes, ci values and labels
            (160xcount+3)

    Returns:
        None
    """
    print("Saving "+str(count)+" extracted features")
    columns = []
    for step in range(1, count+1):
        columns.append("fe-"+str(step))

    columns.append("CIHigh")
    columns.append("CILow")
    columns.append("MOS")

    pca_features_file_name = filename_prefix+str(count)+".csv"

    df = pd.DataFrame(attributes_ci_labels_as_array, columns=columns)
    df.to_csv(pca_features_file_name, index=False)


def generate_features(attributes, labels, n_features, method):
    """
    Extracts features using the method name provide and saves to respective
    files in CSV format.

    Arguments:
        attributes -- attributes with confidence intervals, list(160x127)
        labels -- labels, list(160,)
        n_features -- int
        method -- pca, incremental_pca, fast_ica or kernel_pca

    Returns:
        None
    """

    attributes_as_array, ci_labels_as_array = \
        prepare_data_for_feature_extraction(attributes, labels)

    for count in range(1, n_features+1):

        if method == 'pca':
            fe_method = PCA(n_components=count)
            fileprefix = PCA_FEATURES_FILE_PREFIX
        elif method == 'fast_ica':
            fe_method = FastICA(n_components=count)
            fileprefix = FAST_ICA_FEATURES_FILE_PREFIX
        elif method == 'incremental_pca':
            fe_method = IncrementalPCA(n_components=count)
            fileprefix = INCREMENTAL_PCA_FEATURES_FILE_PREFIX
        elif method == 'kernel_pca':
            fe_method = KernelPCA(n_components=count)
            fileprefix = KERNEL_PCA_FEATURES_FILE_PREFIX

        # 160 x count
        attributes_new = fe_method.fit_transform(attributes_as_array)
        # print(attributes_new.shape)

        # count features, CIHigh, CILow, MOS (160 x count+2)
        attributes_ci_labels_as_array = \
            concatenate((attributes_new, ci_labels_as_array), axis=1)
        # print(attributes_ci_labels_as_array.shape)

        save_extracted_features(fileprefix, count,
                                attributes_ci_labels_as_array)
