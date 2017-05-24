# ML2017 final
# DengAI: Predicting Disease Spread
# Neural Network

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import csv
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

def main():
    
    print('=================================================================')
    print('Read data')
    (X_train_sj, X_train_iq), (_, _) = read_features('dengue_features_train.csv')
    (X_test_sj, X_test_iq), (index_sj, index_iq) = read_features('dengue_features_test.csv')
    (Y_train_sj, Y_train_iq) = read_labels('dengue_labels_train.csv')
    print('Shape of train_sj:', X_train_sj.shape, ', train_iq:', X_train_iq.shape)
    print('Shape of test_sj:', X_test_sj.shape, ', test_iq:', X_test_iq.shape)

    print('Filter bad data')
    #index = np.where(Y_train_sj < 1000)[0]
    #X_train_sj = X_train_sj[index]
    #Y_train_sj = Y_train_sj[index]

    print('=================================================================')
    print('Interpolation')
    X_train_sj = interpolation(X_train_sj)
    X_train_iq = interpolation(X_train_iq)
    X_test_sj = interpolation(X_test_sj)
    X_test_iq = interpolation(X_test_iq)

    print('=================================================================')
    print('Normalization')
    X_train_sj = normalization(X_train_sj)
    X_train_iq = normalization(X_train_iq)
    X_test_sj = normalization(X_test_sj)
    X_test_iq = normalization(X_test_iq)

    print('=================================================================')
    print('Split validation')
    (val_sj, val_iq) = (100, 50)

    X_train_sj_val = X_train_sj[:val_sj]
    X_train_iq_val = X_train_iq[:val_iq]
    Y_train_sj_val = Y_train_sj[:val_sj]
    Y_train_iq_val = Y_train_iq[:val_iq]

    X_train_sj = X_train_sj[val_sj:]
    X_train_iq = X_train_iq[val_iq:]
    Y_train_sj = Y_train_sj[val_sj:]
    Y_train_iq = Y_train_iq[val_iq:]


    print('=================================================================')
    EPOCHS = 500
    
    print('Construct model (sj)')
    dim = X_train_sj.shape[1]
    size = X_train_sj.shape[0]

    model_sj = Sequential()
    model_sj.add(Dense(128, input_shape=(dim,), activation='linear'))
    model_sj.add(Dense(256, activation='linear'))
    model_sj.add(Dense(512, activation='elu'))
    model_sj.add(Dense(512, activation='elu'))
    model_sj.add(Dense(512, activation='elu'))
    model_sj.add(Dense(256, activation='linear'))
    model_sj.add(Dense(128, activation='linear'))
    model_sj.add(Dense(64, activation='linear'))
    model_sj.add(Dense(1, activation='linear'))

    print('Compile model')
    model_sj.compile(optimizer='adam', loss='mean_absolute_error')

    print('Train model')
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
    check_point = ModelCheckpoint(filepath='model_sj.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    model_sj.fit(X_train_sj, Y_train_sj, \
                 batch_size=size, epochs=EPOCHS, verbose=2, \
                 validation_data=(X_train_sj_val, Y_train_sj_val), \
                 callbacks=[early_stop, check_point])

    model_sj = load_model('model_sj.h5')


    print('-----------------------------------------------------------------')
    print('Construct model (iq)')
    dim = X_train_iq.shape[1]
    size = X_train_iq.shape[0]

    model_iq = Sequential()
    model_iq.add(Dense(128, input_shape=(dim,), activation='linear'))
    model_iq.add(Dense(256, activation='elu'))
    model_iq.add(Dense(256, activation='elu'))
    model_iq.add(Dense(128, activation='linear'))
    model_iq.add(Dense(1, activation='linear'))

    print('Compile model')
    model_iq.compile(optimizer='adam', loss='mean_absolute_error')

    print('Train model')
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
    check_point = ModelCheckpoint(filepath='model_iq.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    model_iq.fit(X_train_iq, Y_train_iq, \
                 batch_size=size, epochs=EPOCHS, verbose=2, \
                 validation_data=(X_train_iq_val, Y_train_iq_val), \
                 callbacks=[early_stop, check_point])

    model_iq = load_model('model_iq.h5')
    
    print('=================================================================')
    print('Predict')
    result_sj = model_sj.predict(X_test_sj, verbose=1)
    result_iq = model_iq.predict(X_test_iq, verbose=1)
    result = np.concatenate((result_sj, result_iq), axis=0)
    index = np.concatenate((index_sj, index_iq), axis=0)
    output_result('NN_result.csv', index, result)


if __name__ == '__main__':
    main()
