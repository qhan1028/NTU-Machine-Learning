# ML 2017 hw6
# Matrix Factorization (Train)

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import csv
import numpy as np
import keras.backend as K
from keras.layers import Input, Embedding, Flatten, Dense, Dropout
from keras.layers.merge import Dot, Add, Concatenate
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
    print('Read Data')
    movies, all_genres = read_movie(DATA_DIR + '/movies.csv')
    genders, ages, occupations = read_user(DATA_DIR + '/users.csv')
    train = read_train(DATA_DIR + '/train.csv')

    print('============================================================')
    print('Preprocess Data')
    userID, movieID, userGender, userAge, userOccu, movieGenre, Y = \
        preprocess(train, genders, ages, occupations, movies)
    #ratingMean = find_avg_Y(train)
    #userAvgY = np.array(ratingMean)[userID]

    n_users = np.max(userID) + 1
    n_movies = np.max(movieID) + 1

    print('============================================================')
    print('Construct Model')
    EMB_DIM = 128
    print('Embedding Dimension:', EMB_DIM)
    # inputs
    in_userID = Input(shape=(1,), name='in_userID')       # user id
    in_movieID = Input(shape=(1,), name='in_movieID')      # movie id
    #in_userAvgY = Input(shape=(1,), name='in_userAvgY')
    # embeddings
    emb_userID = Embedding(n_users, EMB_DIM, name='emb_userID')(in_userID)
    emb_movieID = Embedding(n_movies, EMB_DIM, name='emb_movieID')(in_movieID)
    vec_userID = Dropout(0.5)( Flatten(name='vec_userID')(emb_userID) )
    vec_movieID = Dropout(0.5)( Flatten(name='vec_movieID')(emb_movieID) )
    # dot
    dot = Dot(axes=1)([vec_userID, vec_movieID])
    # bias
    emb2_userID = Embedding(n_users, 1, name='emb2_userID')(in_userID)
    emb2_movieID = Embedding(n_movies, 1, name='emb2_movieID')(in_movieID)
    bias_userID = Flatten(name='bias_userID')(emb2_userID)
    bias_movieID = Flatten(name='bias_movieID')(emb2_movieID)
    # output 
    out = Add()([bias_userID, bias_movieID, dot])
    # model
    model = Model(inputs=[in_userID, in_movieID], outputs=dot)
    model.summary()

    def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )
    model.compile(optimizer='adam', loss='mse', metrics=[rmse])
   
    print('============================================================')
    print('Train Model')
    es = EarlyStopping(monitor='val_rmse', patience=30, verbose=1, mode='min')
    cp = ModelCheckpoint(monitor='val_rmse', save_best_only=True, save_weights_only=False, \
                         mode='min', filepath='mf_simple_model.h5')
    history = model.fit([userID, movieID], Y, \
                        epochs=1000, verbose=1, batch_size=10000, callbacks=[es, cp], \
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
    #userAvgY = np.array(ratingMean)[userID]

    result = model.predict([userID, movieID])

    print('Output Result')
    rating = np.clip(result, 1, 5).reshape(-1, 1)
    output = np.array( np.concatenate((ID, rating), axis=1))
   
    print('============================================================')
    print('Save Result')
    write_result(PRED_DIR + '/mf_simple_' + best_val + '.csv', output)
    np.savez(HIS_DIR + '/mf_simple_' + best_val + '_his.npz', rmse=H['rmse'], val_rmse=H['val_rmse'])
    os.rename('mf_simple_model.h5', MODEL_DIR + '/mf_simple_' + best_val + '.h5')


if __name__ == '__main__':
    main()
