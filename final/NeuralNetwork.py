# ML2017 final
# DengAI: Predicting Disease Spread
# Neural Network

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import csv
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from reader import *


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


def output_result(filename, index, result):

    for i, res in enumerate(result):
        result[i] = round(res[0])
    
    result = np.array(result, dtype='int64')
    output = np.concatenate((index, result), axis=1)

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['city', 'year', 'weekofyear', 'total_cases'])
        writer.writerows(output)

DATA_DIR = './data'
MODEL_DIR = './model'
PRED_DIR = './predict'
HIS_DIR = './history'

def main():
    
    print('\n=================================================================')
    print('Read data')
    (X_train_sj, X_train_iq), (_, _) = read_features(DATA_DIR + '/dengue_features_train.csv')
    (X_test_sj, X_test_iq), (index_sj, index_iq) = read_features(DATA_DIR + '/dengue_features_test.csv')
    (Y_train_sj, Y_train_iq) = read_labels(DATA_DIR + '/dengue_labels_train.csv')
    print('Shape of train_sj:', X_train_sj.shape, ', train_iq:', X_train_iq.shape)
    print('Shape of test_sj:', X_test_sj.shape, ', test_iq:', X_test_iq.shape)

    print('\n=================================================================')
    print('Interpolation')
    X_train_sj = interpolation(X_train_sj)
    X_train_iq = interpolation(X_train_iq)
    X_test_sj = interpolation(X_test_sj)
    X_test_iq = interpolation(X_test_iq)

    print('\n=================================================================')
    print('Normalization')
    X_train_sj = normalization(X_train_sj)
    X_train_iq = normalization(X_train_iq)
    X_test_sj = normalization(X_test_sj)
    X_test_iq = normalization(X_test_iq)

    print('\n=================================================================')
    print('Shuffle')
    np.random.seed(3318)
    shuffle_sj = np.random.permutation(len(X_train_sj))
    X_train_sj, Y_train_sj = X_train_sj[shuffle_sj], Y_train_sj[shuffle_sj]

    shuffle_iq = np.random.permutation(len(X_train_iq))
    X_train_iq, Y_train_iq = X_train_iq[shuffle_iq], Y_train_iq[shuffle_iq]

    print('\n=================================================================')
    epoch_sj, epoch_iq = 1000, 1000
    patience = 500
    print('Construct model (sj)')
    dim = X_train_sj.shape[1]
    size = X_train_sj.shape[0]

    model_sj = Sequential()
    model_sj.add(Dense(256, input_shape=(dim,), activation='relu'))
    model_sj.add(Dropout(0.2))
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
    his = model_sj.fit(X_train_sj, Y_train_sj, \
                       batch_size=size, epochs=epoch_sj, verbose=2, validation_split=0.1, callbacks=[es, cp])

    his_sj = his.history

    print('\n-----------------------------------------------------------------')
    print('Construct model (iq)')
    dim = X_train_iq.shape[1]
    size = X_train_iq.shape[0]

    model_iq = Sequential()
    model_iq.add(Dense(256, input_shape=(dim,), activation='relu'))
    model_iq.add(Dropout(0.2))
    model_iq.add(Dense(512, activation='sigmoid'))
    model_iq.add(Dense(512, activation='elu'))
    model_iq.add(Dense(256, activation='elu'))
    model_iq.add(Dense(128, activation='elu'))
    model_iq.add(Dense(1, activation='linear'))

    print('Compile model')
    model_iq.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print('Train model')
    es = EarlyStopping(monitor='val_mean_absolute_error', mode='min', patience=patience, verbose=1)
    cp = ModelCheckpoint(filepath=MODEL_DIR + '/model_iq.h5', \
                         monitor='val_mean_absolute_error', save_best_only=True, mode='min', verbose=0)
    his = model_iq.fit(X_train_iq, Y_train_iq, \
                       batch_size=size, epochs=epoch_iq, verbose=2, validation_split=0.1, callbacks=[es, cp])

    his_iq = his.history
    
    print('\n=================================================================')
    print('Predict last')
    result_sj = model_sj.predict(X_test_sj)
    result_iq = model_iq.predict(X_test_iq)
    result_last = np.concatenate((result_sj, result_iq), axis=0)
    index = np.concatenate((index_sj, index_iq), axis=0)

    print('Last validation MAE')
    last_val_sj = '{:.4f}'.format(his_sj['val_mean_absolute_error'][-1])
    last_val_iq = '{:.4f}'.format(his_iq['val_mean_absolute_error'][-1])
    print('sj: ' + last_val_sj + '\niq: ' + last_val_iq)

    print('Save last model')
    model_sj.save(MODEL_DIR + '/last_sj_' + last_val_sj + '.h5')
    model_iq.save(MODEL_DIR + '/last_iq_' + last_val_iq + '.h5')

    print('Save last prediction')
    output_result(PRED_DIR + '/last_sj_' + last_val_sj + '_iq_' + last_val_iq + '.csv', index, result_last)

    print('\n=================================================================')
    print('Predict best')
    model_sj = load_model(MODEL_DIR + '/model_sj.h5')
    model_iq = load_model(MODEL_DIR + '/model_iq.h5')
    result_sj = model_sj.predict(X_test_sj)
    result_iq = model_iq.predict(X_test_iq)
    result_best = np.concatenate((result_sj, result_iq), axis=0)
    index = np.concatenate((index_sj, index_iq), axis=0)

    print('Best validation MAE')
    best_val_sj = '{:.4f}'.format(np.min(his_sj['val_mean_absolute_error']))
    best_val_iq = '{:.4f}'.format(np.min(his_iq['val_mean_absolute_error']))
    print('sj: ' + best_val_sj + '\niq: ' + best_val_iq)

    print('Save best model')
    os.rename(MODEL_DIR + '/model_sj.h5', MODEL_DIR + '/sj_' + best_val_sj + '.h5')
    os.rename(MODEL_DIR + '/model_iq.h5', MODEL_DIR + '/iq_' + best_val_iq + '.h5')

    print('Save best prediction')
    output_result(PRED_DIR + '/sj_' + best_val_sj + '_iq_' + best_val_iq + '.csv', index, result_best)

    print('\n=================================================================')
    print('Plot history')
    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(hspace=0.2)

    plt.subplot(2, 1, 1)
    plt.plot(his_sj['mean_absolute_error'], 'b')
    plt.plot(his_sj['val_mean_absolute_error'], 'r')
    plt.grid(linestyle=':')
    plt.ylabel('mean absolute error')
    plt.title('sj')

    plt.subplot(2, 1, 2)
    plt.plot(his_iq['mean_absolute_error'], 'b')
    plt.plot(his_iq['val_mean_absolute_error'], 'r')
    plt.grid(linestyle=':')
    plt.xlabel('epochs')
    plt.ylabel('mean absolute error')
    plt.title('iq')

    plt.savefig(HIS_DIR + '/sj_' + best_val_sj + '_iq_' + best_val_iq + '.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
    if not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)
    if not os.path.exists(PRED_DIR): os.mkdir(PRED_DIR)
    if not os.path.exists(HIS_DIR): os.mkdir(HIS_DIR)
    main()
