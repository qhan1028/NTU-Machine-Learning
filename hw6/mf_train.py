# ML 2017 hw6
# Matrix Factorization (Train)

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import csv
import numpy as np
import keras.backend as K
from keras.layers import Input, Embedding, Flatten, Dense
from keras.layers.merge import dot, add
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from reader import *

DATA_DIR = './data'

def main():
   
    print('============================================================')
    print('Read Data')
    movies, all_genres, n_movies = read_movie(DATA_DIR + '/movies.csv')
    genders, ages, occupations, n_users = read_user(DATA_DIR + '/users.csv')
    train = read_train(DATA_DIR + '/train.csv')
    print('n_movies:', n_movies, ', n_users:', n_users)
    print('Train data len:', len(train))

    print('============================================================')
    print('Shuffle Data')
    index = np.random.permutation(len(train))
    train = train[index]

    print('============================================================')
    print('Get User/Movie ID')
    user_id = np.array(train[:, 1], dtype='int32')
    movie_id = np.array(train[:, 2], dtype='int32')
    
    print('Get User/Movie Features')
    user_genders = np.array(genders)[user_id].reshape(-1, 1)
    user_ages = np.array(ages)[user_id].reshape(-1, 1)
    movie_genres = np.array(movies)[movie_id]
    user_id = user_id.reshape(-1, 1)
    movie_id = movie_id.reshape(-1, 1)

    print('Get Y')
    Y_rating = train[:, 3].reshape(-1, 1)
    print('Y_rating:', Y_rating.shape)
    print()

    n_users = np.max(user_id) + 1
    n_movies = np.max(movie_id) + 1
    n_genders = 2
    n_ages = np.max(user_ages) + 1
    
    print('============================================================')
    print('Construct Model')
    in_uid = Input(shape=(1,), name='user id')      # user id
    in_mid = Input(shape=(1,), name='movie id')     # movie id
    in_ug = Input(shape=(1,), name='user gender')   # user gender
    in_ua = Input(shape=(1,), name='user age')      # user age
    in_mg = Input(shape=(18,), name='movie genre')  # movie genre
    emb_uid = Embedding(input_dim=n_users, output_dim=128, input_length=1)(in_uid)
    emb_mid = Embedding(input_dim=n_movies, output_dim=128, input_length=1)(in_mid)
    emb_ug = Embedding(input_dim=n_genders, output_dim=128, input_length=1)(in_ug)
    emb_ua = Embedding(input_dim=n_ages, output_dim=128, input_length=1)(in_ua)
    fl_uid = Flatten()(emb_uid)
    fl_mid = Flatten()(emb_mid)
    fl_ug = Flatten()(emb_ug)
    fl_ua = Flatten()(emb_ua)

    x = Dense(512, activation='elu')(in_mg)
    x = Dense(256, activation='elu')(x)
    fl_mg = Dense(128, activation='elu')(x)

    dot_id = dot(inputs=[fl_uid, fl_mid], axes=1)
    dot_uid_ug = dot(inputs=[fl_uid, fl_ug], axes=1)
    dot_uid_ua = dot(inputs=[fl_uid, fl_ua], axes=1)
    dot_uid_mg = dot(inputs=[fl_uid, fl_mg], axes=1)
    dot_mid_ug = dot(inputs=[fl_mid, fl_ug], axes=1)
    dot_mid_ua = dot(inputs=[fl_mid, fl_ua], axes=1)
    dot_mid_mg = dot(inputs=[fl_mid, fl_mg], axes=1)
    dot_ug_ua = dot(inputs=[fl_ug, fl_ua], axes=1)
    dot_ug_mg = dot(inputs=[fl_ug, fl_mg], axes=1)
    dot_ua_mg = dot(inputs=[fl_ua, fl_mg], axes=1)

    sum_dot = add(inputs=[dot_id, dot_uid_ug, dot_uid_ua, dot_uid_mg, \
                          dot_mid_ug, dot_mid_ua, dot_mid_mg, dot_ug_ua, \
                          dot_ug_mg, dot_ua_mg])

    bu = Embedding(input_dim=n_users, output_dim=1, input_length=1)(in_uid)
    bm = Embedding(input_dim=n_movies, output_dim=1, input_length=1)(in_mid)
    fl_bu = Flatten()(bu)
    fl_bm = Flatten()(bm)
    sum_bias = add(inputs=[fl_bu, fl_bm])

    sum_layer = add(inputs=[sum_dot, sum_bias])

    model = Model(inputs=[in_uid, in_mid, in_ug, in_ua, in_mg], outputs=sum_layer)
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
    history = model.fit([user_id, movie_id, user_genders, user_ages, movie_genres], Y_rating, \
                        epochs=50, verbose=1, batch_size=10000, validation_split=0.1, callbacks=[es, cp])
    H = history.history
   
    print('============================================================')
    print('Save Result')
    np.savez('mf_history.npz', rmse=H['rmse'])


if __name__ == '__main__':
    main()
