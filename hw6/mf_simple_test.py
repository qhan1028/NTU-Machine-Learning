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

DATA_DIR = sys.argv[1]
OUTPUT_FILE = sys.argv[2]
MODEL = sys.argv[3]

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

    print('============================================================')
    print('Preprocess Data')
    test = read_test(DATA_DIR + '/test.csv')
    ID = np.array(test[:, 0]).reshape(-1, 1)
    print('Test data len:', len(test))
    
    userID, movieID, userGender, userAge, userOccu, movieGenre, _Y = \
        preprocess(test, genders, ages, occupations, movies)

    print('============================================================')
    print('Load Model')
    def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )
    model = load_model(MODEL, custom_objects={'rmse': rmse})

    print('============================================================')
    print('Test Model')
    result = model.predict([userID, movieID])

    print('============================================================')
    print('Output Result')
    rating = np.clip(result, 1, 5)
    output = np.array( np.concatenate((ID, rating), axis=1))
    write_result(OUTPUT_FILE, output)


if __name__ == '__main__':
    main()
