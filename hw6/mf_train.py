# ML 2017 hw6
# Matrix Factorization

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import csv
import numpy as np
from keras.layers import Input, Embedding, Flatten
from keras.layers.merge import dot
from keras.layers import Reshape, Embedding, Dropout, Dense, Merge
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from reader import *

def main():
   
    print('============================================================')
    print('Read Data')
    movies, all_genres, movies_len = read_movie('movies.csv')
    users, users_len = read_user('users.csv')
    train = read_train('train.csv')
    test = read_test('test.csv')
    print('Movie len:', movies_len)
    print('User len:', users_len)
    print('Train data len:', len(train))
    print('Test data len:', len(test))
    X_user = np.array(train[:, 1], dtype='int32').reshape(-1, 1)
    X_movie = np.array(train[:, 2], dtype='int32').reshape(-1, 1)
    Y = train[:, 3].reshape(-1, 1)
    n_users = np.max(X_user)
    n_movies = np.max(X_movie)
    print('n_users:', n_users)
    print('n_movies:', n_movies)

    print('============================================================')
    print('Construct Model')
    input_user = Input(shape=(1,))
    embedding_user = Embedding(input_dim=n_users+1, output_dim=120, input_length=1)(input_user)
    flatten_user = Flatten()(embedding_user)
    input_movie = Input(shape=(1,))
    embedding_movie = Embedding(input_dim=n_movies+1, output_dim=120, input_length=1)(input_movie)
    flatten_movie = Flatten()(embedding_movie)
    dot_layer = dot(inputs=[flatten_user, flatten_movie], axes=1)

    model = Model(inputs=[input_user, input_movie], outputs=dot_layer)
    model.summary()
    model.compile(optimizer='adamax', loss='mean_squared_error')
   
    print('============================================================')
    print('Train Model')
    history = model.fit([X_user, X_movie], Y, epochs=30, validation_split=0.1, verbose=1)
    H = history.history
    
    model.save('model.h5')


if __name__ == '__main__':
    main()
