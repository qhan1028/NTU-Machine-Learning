# ML 2017 hw6
# Matrix Factorization (Train)

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import csv
import numpy as np
import keras.backend as K
from keras.layers import Input, Embedding, Flatten, Dense
from keras.layers.merge import Dot, Add, Concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from reader import *

DATA_DIR = './data'


def write_result(filename, output):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['TestDataID', 'Rating'])
        writer.writerows(output)


def main():
   
    print('============================================================')
    print('Read Data')
    movies, all_genres = read_movie(DATA_DIR + '/movies.csv')
    genders, ages, occupations = read_user(DATA_DIR + '/users.csv')
    print('movies:', np.array(movies).shape)
    print('genders:', np.array(genders).shape)
    print('ages:', np.array(ages).shape)
    print('occupations:', np.array(occupations).shape)

    train = read_train(DATA_DIR + '/train.csv')
    print('Train data len:', len(train))

    print('============================================================')
    print('Preprocess Data')
    userID, movieID, userGender, userAge, userOccu, movieGenre, Y = \
        preprocess(train, genders, ages, occupations, movies)
    print('userID:', userID.shape)
    print('movieID:', movieID.shape)
    print('userGender:', userGender.shape)
    print('userAge:', userAge.shape)
    print('userOccu:', userOccu.shape)
    print('movieGenre:', movieGenre.shape)
    print('Y:', Y.shape)

    n_users = np.max(userID) + 1
    n_movies = np.max(movieID) + 1

    print('Normalize Rating')
    Y_mean = np.mean(Y)
    Y_std = np.std(Y)
    Y = (Y - Y_mean) / Y_std

    print('============================================================')
    print('Construct Model')
    EMB_DIM = 512
    print('Embedding Dimension:', EMB_DIM)
    # inputs
    in_userID = Input(shape=(1,))       # user id
    in_movieID = Input(shape=(1,))      # movie id
    in_userGender = Input(shape=(1,))   # user gender
    in_userAge = Input(shape=(1,))      # user age
    in_userOccu = Input(shape=(21,))    # user occupation
    in_movieGenre = Input(shape=(18,))  # movie genre
    # embeddings
    emb_userID = Embedding(n_users, EMB_DIM)(in_userID)
    emb_movieID = Embedding(n_movies, EMB_DIM)(in_movieID)
    vec_userID = Flatten()(emb_userID)
    vec_movieID = Flatten()(emb_movieID)
    vec_userOccu = Dense(EMB_DIM, activation='linear')(in_userOccu)
    vec_movieGenre = Dense(EMB_DIM, activation='linear')(in_movieGenre)
    # dot
    dot1 = Dot(axes=1)([vec_userID, vec_movieID])
    dot2 = Dot(axes=1)([vec_userID, vec_userOccu])
    dot3 = Dot(axes=1)([vec_userID, vec_movieGenre])
    dot4 = Dot(axes=1)([vec_movieID, vec_userOccu])
    dot5 = Dot(axes=1)([vec_movieID, vec_movieGenre])
    dot6 = Dot(axes=1)([vec_userOccu, vec_movieGenre])
    # concatenate
    con_dot = Concatenate()([dot1, dot2, dot3, dot4, dot5, dot6, \
                             in_userGender, in_userAge])
    dense_out = Dense(1, activation='linear')(con_dot)
    # bias
    emb2_userID = Embedding(n_users, 1, embeddings_initializer='zeros')(in_userID)
    emb2_movieID = Embedding(n_movies, 1, embeddings_initializer='zeros')(in_movieID)
    bias_userID = Flatten()(emb2_userID)
    bias_movieID = Flatten()(emb2_movieID)
    # output 
    out = Add()([bias_userID, bias_movieID, dot1])
    # model (userID * movieID only)
    model = Model(inputs=[in_userID, in_movieID], outputs=dot1)
    model.summary()

    def rmse(y_true, y_pred):
        return K.sqrt( K.mean(((y_pred - y_true) * Y_std)**2) )
    model.compile(optimizer='adam', loss='mse', metrics=[rmse])
   
    print('============================================================')
    print('Train Model')
    es = EarlyStopping(monitor='val_rmse', patience=10, verbose=1, mode='min')
    cp = ModelCheckpoint(monitor='val_rmse', save_best_only=True, save_weights_only=False, \
                         mode='min', filepath='mf_simple_model.h5')
    history = model.fit([userID, movieID], Y, \
                        epochs=500, verbose=1, batch_size=10000, callbacks=[es, cp], \
                        validation_split=0.05)
    H = history.history
    best_val = str( round(np.min(H['val_rmse']), 6) )
    print('Best Val RMSE:', best_val)

    print('============================================================')
    print('Test Model')
    model = load_model('mf_simple_model.h5', custom_objects={'rmse': rmse})
    test = read_test(DATA_DIR + '/test.csv')
    ID = np.array(test[:, 0]).reshape(-1, 1)
    print('Test data len:', len(test))
    
    userID, movieID, userGender, userAge, userOccu, movieGenre, _Y = \
        preprocess(test, genders, ages, occupations, movies)

    print('Un-Normalize')
    result = model.predict([userID, movieID])
    result = result * Y_std + Y_mean

    print('Output Result')
    rating = np.clip(result, 1, 5).reshape(-1, 1)
    output = np.array( np.concatenate((ID, rating), axis=1))
    print(output[:20])
   
    print('============================================================')
    print('Save Result')
    write_result('mf_simple_norm.csv', output)
    np.savez('mf_simple_' + best_val + '_his.npz', rmse=H['rmse'], val_rmse=H['val_rmse'])
    os.rename('mf_simple_model.h5', 'mf_simple_' + best_val + '.h5')


if __name__ == '__main__':
    main()
