# ML 2017 hw6
# Deep Neural Network (Train)

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import csv
import numpy as np
import keras.backend as K
from keras.layers import Input, Embedding, Flatten, Dense, Dropout
from keras.layers.merge import Concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from reader import *

DATA_DIR = './data'
MODEL_DIR = './model'
PRED_DIR = './predict'
HIS_DIR = './history'

def write_result(filename, output):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['TestDataID', 'Rating'])
        writer.writerows(output)


def main():
   
    print('============================================================')
    print('Read Data', flush=True)
    movies, all_genres = read_movie(DATA_DIR + '/movies.csv')
    genders, ages, occupations = read_user(DATA_DIR + '/users.csv')
    train = read_train(DATA_DIR + '/train.csv')

    print('============================================================')
    print('Preprocess Data', flush=True)
    userID, movieID, userGender, userAge, userOccu, movieGenre, Y = \
        preprocess(train, genders, ages, occupations, movies)

    n_users = np.max(userID) + 1
    n_movies = np.max(movieID) + 1

    print('============================================================')
    print('Construct Model')
    DIM = 128
    print('Embedding Dimension:', DIM)
    # input
    in_userID = Input(shape=(1,))
    in_movieID = Input(shape=(1,))
    in_userGender = Input(shape=(1,))
    in_userAge = Input(shape=(1,))
    in_userOccu = Input(shape=(21,))
    in_movieGenre = Input(shape=(18,))
    # embedding
    emb_userID = Embedding(n_users, DIM)(in_userID)
    emb_movieID = Embedding(n_movies, DIM)(in_movieID)
    vec_userID = Dropout(0.5)( Flatten()(emb_userID) )
    vec_movieID = Dropout(0.5)( Flatten()(emb_movieID) )
    vec_userOccu = Dropout(0.5)( Dense(DIM, activation='linear')(in_userOccu) )
    vec_movieGenre = Dropout(0.5)( Dense(DIM, activation='linear')(in_movieGenre) )
    # concatenate
    x = Concatenate()([vec_userID, vec_movieID, vec_userOccu, vec_movieGenre, \
                       in_userGender, in_userAge])
    # dense
    x = Dense(128, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    x = Dense(32, activation='elu')(x)
    # output
    out = Dense(1, activation='linear')(x)
    # model
    model = Model(inputs=[in_userID, in_movieID, in_userGender, in_userAge, in_userOccu, in_movieGenre], outputs=out)
    model.summary()

    def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )

    model.compile(optimizer='adam', loss='mse', metrics=[rmse])
   
    print('============================================================')
    print('Train Model')
    es = EarlyStopping(monitor='val_rmse', patience=30, verbose=1, mode='min')
    cp = ModelCheckpoint(monitor='val_rmse', save_best_only=True, save_weights_only=False, \
                         mode='min', filepath='dnn_model.h5')
    history = model.fit([userID, movieID, userGender, userAge, userOccu, movieGenre], Y, \
                        epochs=200, verbose=1, batch_size=10000, callbacks=[es, cp], \
                        validation_split=0.05)
    H = history.history
    best_val = str( round(np.min(H['val_rmse']), 6) )
    print('Best Val:', best_val)

    print('============================================================')
    print('Test Model')
    model = load_model('dnn_model.h5', custom_objects={'rmse': rmse})
    test = read_test(DATA_DIR + '/test.csv')
    ID = np.array(test[:, 0]).reshape(-1, 1)
    print('Test data len:', len(test))
    
    userID, movieID, userGender, userAge, userOccu, movieGenre, _Y = \
        preprocess(test, genders, ages, occupations, movies)

    result = model.predict([userID, movieID, userGender, userAge, userOccu, movieGenre])

    print('Output Result')
    rating = np.clip(result, 1, 5).reshape(-1, 1)
    output = np.array( np.concatenate((ID, rating), axis=1))
    print(output[:20])
   
    print('============================================================')
    print('Save Result')
    write_result(PRED_DIR + '/dnn_' + best_val + '.csv', output)
    np.savez(HIS_DIR + '/dnn_' + best_val + '_his.npz', rmse=H['rmse'], val_rmse=H['val_rmse'])
    os.rename('dnn_model.h5', MODEL_DIR + '/dnn_' + best_val + '.h5')

if __name__ == '__main__':
    main()
