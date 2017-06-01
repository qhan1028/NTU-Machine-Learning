# ML 2017 hw6
# Matrix Factorization (Train)

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import csv
import numpy as np
import keras.backend as K
from keras.layers import Input, Embedding, Flatten, Dense
from keras.layers.merge import dot, add, concatenate
from keras.models import Model, load_model
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
    np.random.seed(2048)
    index = np.random.permutation(len(train))
    train = train[index]

    print('============================================================')
    print('Get User/Movie ID')
    user_id = np.array(train[:, 1], dtype=int)
    movie_id = np.array(train[:, 2], dtype=int)
    
    print('Get User/Movie Features')
    user_genders = np.array(genders)[user_id].reshape(-1, 1)
    user_ages = np.array(ages)[user_id].reshape(-1, 1)
    movie_genres = np.array(movies)[movie_id]
    user_id = user_id.reshape(-1, 1)
    movie_id = movie_id.reshape(-1, 1)

    print('Get Y')
    Y_rating = train[:, 3].reshape(-1, 1)
    print('Y_rating:', Y_rating.shape)

    n_users = np.max(user_id) + 1
    n_movies = np.max(movie_id) + 1
    n_genders = 2
    n_ages = np.max(user_ages) + 1
    
    print('============================================================')
    print('Construct Model')
    EMB_DIM = 128
    print('Embedding Dimension:', EMB_DIM)
    in_uid = Input(shape=[1], name='UserID')      # user id
    in_mid = Input(shape=[1], name='MovieID')     # movie id
    in_ug = Input(shape=[1], name='UserGender')   # user gender
    in_ua = Input(shape=[1], name='UserAge')      # user age
    in_mg = Input(shape=[18], name='MovieGenre')  # movie genre
    emb_uid = Embedding(n_users, EMB_DIM, embeddings_initializer='random_normal')(in_uid)
    emb_mid = Embedding(n_movies, EMB_DIM, embeddings_initializer='random_normal')(in_mid)
    emb_ug = Embedding(n_genders, EMB_DIM, embeddings_initializer='random_normal')(in_ug)
    emb_ua = Embedding(n_ages, EMB_DIM, embeddings_initializer='random_normal')(in_ua)
    fl_uid = Flatten()(emb_uid)
    fl_mid = Flatten()(emb_mid)
    fl_ug = Flatten()(emb_ug)
    fl_ua = Flatten()(emb_ua)

    fl_mg = Dense(EMB_DIM, activation='linear', name='MovieGenre_dense')(in_mg)

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

    con_dot = concatenate(inputs=[dot_id, dot_uid_mg, \
                                  dot_mid_ug, dot_mid_ua, \
                                  dot_ug_mg, dot_ua_mg])
    
    dense_dot = Dense(1, activation='linear')(con_dot)

    emb_uid = Embedding(n_users, 1, embeddings_initializer='zeros')(in_uid)
    emb_mid = Embedding(n_movies, 1, embeddings_initializer='zeros')(in_mid)
    emb_ug = Embedding(n_genders, 1, embeddings_initializer='zeros')(in_ug)
    emb_ua = Embedding(n_ages, 1, embeddings_initializer='zeros')(in_ua)
    bias_uid = Flatten()(emb_uid)
    bias_mid = Flatten()(emb_mid)
    bias_ug = Flatten()(emb_ug)
    bias_ua = Flatten()(emb_ua)
    
    out = add(inputs=[bias_uid, bias_mid, bias_ug, bias_ua, dense_dot])

    model = Model(inputs=[in_uid, in_mid, in_ug, in_ua, in_mg], outputs=out)
    model.summary()

    def rmse(y_true, y_pred):
        mse = K.mean((y_pred - y_true) ** 2)
        return K.sqrt(mse)

    model.compile(optimizer='adam', loss='mse', metrics=[rmse])
   
    print('============================================================')
    print('Train Model')
    es = EarlyStopping(monitor='val_rmse', patience=10, verbose=1, mode='min')
    cp = ModelCheckpoint(monitor='val_rmse', save_best_only=True, save_weights_only=False, \
                         mode='min', filepath='mf_model.h5')
    history = model.fit([user_id, movie_id, user_genders, user_ages, movie_genres], Y_rating, \
                        epochs=200, verbose=1, batch_size=10000, validation_split=0.1, callbacks=[es, cp])
    H = history.history
   
    print('============================================================')
    print('Evaluate Model')
    model = load_model('mf_model.h5', custom_objects={'rmse': rmse})
    score = model.evaluate([user_id, movie_id, user_genders, user_ages, movie_genres], Y_rating, \
                           batch_size=10000)
    print('Score:', score)

    print('============================================================')
    print('Save Result')
    np.savez('mf_history.npz', rmse=H['rmse'])


if __name__ == '__main__':
    main()
