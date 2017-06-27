# ML2017 final
# DengAI: Predicting Disease Spread
# Neural Network

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import csv
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Embedding, Flatten
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Reader import *
from Preprocess import *
from PlotModel import *

def output_result(filename, index, result):

    for i, res in enumerate(result):
        result[i] = round(res[0])
    
    result = np.array(result, dtype='int64')
    output = np.concatenate((index, result), axis=1)

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['city', 'year', 'weekofyear', 'total_cases'])
        writer.writerows(output)

# argv: [1]data directory
DATA_DIR = sys.argv[1]
MODEL_DIR = '../model'
PRED_DIR = '../predict'
HIS_DIR = '../history'

def main():
    
    print('\n=================================================================')
    print('Read data')
    (X_sj, X_iq), (_, _), (W_sj, W_iq) = read_features(DATA_DIR + '/dengue_features_train.csv')
    (Xt_sj, Xt_iq), (I_sj, I_iq), (Wt_sj, Wt_iq) = read_features(DATA_DIR + '/dengue_features_test.csv')
    (Y_sj, Y_iq) = read_labels(DATA_DIR + '/dengue_labels_train.csv')
    print('train_sj:', X_sj.shape, '\ntrain_iq:', X_iq.shape)
    print('test_sj:', Xt_sj.shape, '\ntest_iq:', Xt_iq.shape)

    print('Interpolation')
    X_sj = interpolation(X_sj)
    X_iq = interpolation(X_iq)
    Xt_sj = interpolation(Xt_sj)
    Xt_iq = interpolation(Xt_iq)

    print('Normalization')
    X_sj = normalization(X_sj)
    X_iq = normalization(X_iq)
    Xt_sj = normalization(Xt_sj)
    Xt_iq = normalization(Xt_iq)

    print('Shuffle')
    X_sj, Y_sj = shuffle(X_sj, Y_sj, 726)
    X_iq, Y_iq = shuffle(X_iq, Y_iq, 1019)

    print('\n=================================================================')
    epoch_sj, epoch_iq = 1000, 1000
    patience = 300
    size, dim = X_sj.shape

    print('Construct model (sj)')
    model_sj = Sequential()
    model_sj.add(Dense(256, input_shape=(dim,), activation='linear'))
    model_sj.add(Dense(256, activation='sigmoid'))
    model_sj.add(Dense(512, activation='elu'))
    model_sj.add(Dense(512, activation='elu'))
    model_sj.add(Dense(256, activation='elu'))
    model_sj.add(Dense(128, activation='elu'))
    model_sj.add(Dense(64, activation='elu'))
    model_sj.add(Dense(1, activation='linear'))

    print('Compile model')
    model_sj.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print('Train model')
    es = EarlyStopping(monitor='val_mean_absolute_error', mode='min', patience=patience, verbose=1)
    cp = ModelCheckpoint(filepath=MODEL_DIR + '/model_sj.h5', \
                         monitor='val_mean_absolute_error', save_best_only=True, mode='min', verbose=0)
    his = model_sj.fit(X_sj, Y_sj, \
                       batch_size=size, epochs=epoch_sj, verbose=2, validation_split=0.1, callbacks=[es, cp])

    his_sj = his.history

    print('\n-----------------------------------------------------------------')
    size, dim = X_iq.shape

    print('Construct model (iq)')
    model_iq = Sequential()
    model_iq.add(Dense(128, input_shape=(dim,), activation='linear'))
    model_iq.add(Dense(128, activation='sigmoid'))
    model_iq.add(Dense(256, activation='elu'))
    model_iq.add(Dense(128, activation='elu'))
    model_iq.add(Dense(64, activation='elu'))
    model_iq.add(Dense(1, activation='linear'))

    print('Compile model')
    model_iq.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print('Train model')
    es = EarlyStopping(monitor='val_mean_absolute_error', mode='min', patience=patience, verbose=1)
    cp = ModelCheckpoint(filepath=MODEL_DIR + '/model_iq.h5', \
                         monitor='val_mean_absolute_error', save_best_only=True, mode='min', verbose=0)
    his = model_iq.fit(X_iq, Y_iq, \
                       batch_size=size, epochs=epoch_iq, verbose=2, validation_split=0.1, callbacks=[es, cp])

    his_iq = his.history
    
    print('\n=================================================================')
    print('Predict last')
    result_sj = model_sj.predict(Xt_sj)
    result_iq = model_iq.predict(Xt_iq)
    result_last = np.concatenate((result_sj, result_iq), axis=0)
    index = np.concatenate((I_sj, I_iq), axis=0)

    print('Last validation MAE')
    last_val_sj = '{:.4f}'.format(his_sj['val_mean_absolute_error'][-1])
    last_val_iq = '{:.4f}'.format(his_iq['val_mean_absolute_error'][-1])
    print('sj: ' + last_val_sj + '\niq: ' + last_val_iq)

    print('Save last model')
    model_sj.save(MODEL_DIR + '/sj_' + last_val_sj + '_last.h5')
    model_iq.save(MODEL_DIR + '/iq_' + last_val_iq + '_last.h5')

    print('Save last prediction')
    output_result(PRED_DIR + '/sj_' + last_val_sj + '_iq_' + last_val_iq + '_last.csv', index, result_last)

    print('Plot ground truth (last)')
    sj_pred, sj_gt = np.reshape(model_sj.predict(X_sj), -1), np.reshape(Y_sj, -1)
    iq_pred, iq_gt = np.reshape(model_iq.predict(X_iq), -1), np.reshape(Y_iq, -1)
    plot_gt(sj_pred, sj_gt, iq_pred, iq_gt, last_val_sj, last_val_iq)

    print('\n=================================================================')
    print('Predict best')
    model_sj = load_model(MODEL_DIR + '/model_sj.h5')
    model_iq = load_model(MODEL_DIR + '/model_iq.h5')
    result_sj = model_sj.predict(Xt_sj)
    result_iq = model_iq.predict(Xt_iq)
    result_best = np.concatenate((result_sj, result_iq), axis=0)
    index = np.concatenate((I_sj, I_iq), axis=0)

    print('Best validation MAE')
    best_val_sj = '{:.4f}'.format(np.min(his_sj['val_mean_absolute_error']))
    best_val_iq = '{:.4f}'.format(np.min(his_iq['val_mean_absolute_error']))
    print('sj: ' + best_val_sj + '\niq: ' + best_val_iq)

    print('Rename best model')
    os.rename(MODEL_DIR + '/model_sj.h5', MODEL_DIR + '/sj_' + best_val_sj + '.h5')
    os.rename(MODEL_DIR + '/model_iq.h5', MODEL_DIR + '/iq_' + best_val_iq + '.h5')

    print('Save best prediction')
    output_result(PRED_DIR + '/sj_' + best_val_sj + '_iq_' + best_val_iq + '.csv', index, result_best)

    print('Plot ground truth (best)')
    sj_pred, sj_gt = np.reshape(model_sj.predict(X_sj), -1), np.reshape(Y_sj, -1)
    iq_pred, iq_gt = np.reshape(model_iq.predict(X_iq), -1), np.reshape(Y_iq, -1)
    plot_gt(sj_pred, sj_gt, iq_pred, iq_gt, best_val_sj, best_val_iq)

    print('\n=================================================================')
    print('Plot history')
    plot_history(his_sj, his_iq, best_val_sj, best_val_iq)


if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)
    if not os.path.exists(PRED_DIR): os.mkdir(PRED_DIR)
    if not os.path.exists(HIS_DIR): os.mkdir(HIS_DIR)
    main()
