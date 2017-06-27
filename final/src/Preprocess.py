# ML2017 final
# DengAI: Predicting Disease Spread
# Preprocess

import numpy as np


def interpolation(data):
    
    full_index = np.array(range(data.shape[0]))

    for column in range(data.shape[1]):
        feature = data[:, column]
        existed_index = np.where(feature != np.inf)[0]
        existed_feature = feature[existed_index]
        data[:, column] = np.interp(full_index, existed_index, existed_feature)
    
    return data


def normalization(data):
    
    for column in range(data.shape[1]):
        feature = data[:, column]
        mean = feature.mean()
        std = feature.std()
        data[:, column] = (feature - mean) / std
    
    return data


def shuffle(X, Y, seed):

    np.random.seed(seed)
    index = np.random.permutation(len(X))
    X, Y = X[index], Y[index]
    return X, Y
