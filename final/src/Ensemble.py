# ML2017 final
# DengAI: Predicting Disease Spread
# Ensemble

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import csv
from keras.models import load_model
from Reader import *
from Preprocess import *

# argv: [1]data directory
DATA_DIR = sys.argv[1]
MODEL_DIR = '../model'
PRED_DIR = '../predict'
HIS_DIR = '../history'

SJ_LIST = 'sj_list.txt'
IQ_LIST = 'iq_list.txt'

def read_list(filename):

    weights, names = [], []

    with open(filename, 'r') as f:
        for line in f:
            row = line.split()
            weight, name = float(row[0]), row[1]
            weights.append(weight)
            names.append(name)

    return weights, names


def predict(X, W, N):
    
    sum_result = np.zeros( (len(X), 1) )
    sum_weights = 0
    
    for (w, name) in zip(W, N):
        print('weight: %f' % w + ', name: ' + name, end=' ', flush=True)
        model = load_model(MODEL_DIR + '/' + name)
        print('[loaded]', end=' ', flush=True)
        result = model.predict(X)
        sum_weights += w
        sum_result += result * w
        print('[predicted]', flush=True)
    
    return sum_result / sum_weights


def output_result(filename, index, result):

    for i, res in enumerate(result):
        result[i] = round(res[0])
    
    result = np.array(result, dtype='int64')
    output = np.concatenate((index, result), axis=1)

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['city', 'year', 'weekofyear', 'total_cases'])
        writer.writerows(output)


def main():
   
    print('\n=================================================================')
    print('Read data')
    (Xt_sj, Xt_iq), (I_sj, I_iq), (Wt_sj, Wt_iq) = read_features(DATA_DIR + '/dengue_features_test.csv')
    print('test_sj:', Xt_sj.shape, '\ntest_iq:', Xt_iq.shape)

    print('Interpolation')
    Xt_sj = interpolation(Xt_sj)
    Xt_iq = interpolation(Xt_iq)

    print('Normalization')
    Xt_sj = normalization(Xt_sj)
    Xt_iq = normalization(Xt_iq)

    print('Read model list')
    W_sj, N_sj = read_list(SJ_LIST)
    W_iq, N_iq = read_list(IQ_LIST)
    print('sj models: %d\niq models: %d' % (len(W_sj), len(W_iq)))

    print('\n=================================================================')
    print('Predict')
    
    result_sj = predict(Xt_sj, W_sj, N_sj)
    result_iq = predict(Xt_iq, W_iq, N_iq)

    result = np.concatenate((result_sj, result_iq), axis=0)
    index = np.concatenate((I_sj, I_iq), axis=0)
    
    print('\n=================================================================')
    print('Output result')
    output_result(PRED_DIR + '/vote.csv', index, result)


if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)
    if not os.path.exists(PRED_DIR): os.mkdir(PRED_DIR)
    if not os.path.exists(HIS_DIR): os.mkdir(HIS_DIR)
    main()
