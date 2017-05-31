# ML 2017 hw6
# Deep Neural Network (Train)

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import csv
import numpy as np
import keras.backend as K
from keras.layers import Input, Embedding, Flatten, Dense
from keras.layers.merge import concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from reader import *

DATA_DIR = './data'

def main():
   
    print('============================================================')
    print('Read Data')
    movies, all_genres, movies_len = read_movie(DATA_DIR + '/movies.csv')
    users, users_len = read_user(DATA_DIR + '/users.csv')
    train = read_train(DATA_DIR + '/train.csv')
    print('Movies:', movies_len, '\nUsers:', users_len, '\nTrain data:', len(train))

    print('============================================================')
    print('Shuffle Data')
    index = np.random.permutation(len(train))
    train = train[index]

    print('============================================================')
    print('Get X, Y')
    X_user = np.array(train[:, 1], dtype='int32').reshape(-1, 1)
    X_movie = np.array(train[:, 2], dtype='int32').reshape(-1, 1)
    Y = train[:, 3].reshape(-1, 1)
    dim_u, dim_m = np.max(X_user), np.max(X_movie)
    print('dim_u:', dim_u, '\ndim_m:', dim_m)

    print('============================================================')
    print('Construct Model')
    in_u = Input(shape=(1,))
    in_m = Input(shape=(1,))
    emb_u = Embedding(input_dim=dim_u+1, output_dim=256, input_length=1)(in_u)
    emb_m = Embedding(input_dim=dim_m+1, output_dim=256, input_length=1)(in_m)
    fl_u = Flatten()(emb_u)
    fl_m = Flatten()(emb_m)
    x = concatenate(inputs=[fl_u, fl_m])
    x = Dense(256, activation='elu')(x)
    x = Dense(128, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    out = Dense(1, activation='linear')(x)

    model = Model(inputs=[in_u, in_m], outputs=out)
    model.summary()

    def rmse(y_true, y_pred):
        mse = K.mean((y_pred - y_true) ** 2)
        return K.sqrt(mse)

    model.compile(optimizer='rmsprop', loss='mse', metrics=[rmse])
   
    print('============================================================')
    print('Train Model')
    es = EarlyStopping(monitor='val_rmse', patience=5, verbose=1, mode='min')
    cp = ModelCheckpoint(monitor='val_rmse', save_best_only=True, save_weights_only=False, \
                         mode='min', filepath='mf_model.h5')
    history = model.fit([X_user, X_movie], Y, epochs=50, verbose=1, \
                        batch_size=10000, validation_split=0.1, callbacks=[es, cp])
    H = history.history
   
    print('============================================================')
    print('Save Result')
    np.savez('mf_history.npz', rmse=H['rmse'])


if __name__ == '__main__':
    main()
