# ML 2017 hw6
# Matrix Factorization (Test)

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import csv
import numpy as np
import keras.backend as K
from keras.models import load_model
from reader import *

def write_result(filename, output):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['TestDataID', 'Rating'])
        writer.writerows(output)

DATA_DIR = sys.argv[1]
OUTPUT_FILE = sys.argv[2]

# argv: [1]data directory [2]prediction file
def main():
   
    print('============================================================')
    print('Read Data')
    movies, all_genres = read_movie(DATA_DIR + '/movies.csv')
    genders, ages, occupations = read_user(DATA_DIR + '/users.csv')
    print('movies:', np.array(movies).shape)
    print('genders:', np.array(genders).shape)
    print('ages:', np.array(ages).shape)
    print('occupations:', np.array(occupations).shape)

    test = read_test(DATA_DIR + '/test.csv')
    ID = np.array(test[:, 0]).reshape(-1, 1)
    print('Test data len:', len(test))

    print('============================================================')
    print('Preprocess Data')
    userID, movieID, userGender, userAge, userOccu, movieGenre, _Y = \
        preprocess(test, genders, ages, occupations, movies)

    print('============================================================')
    print('Load Model')

    def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )

    model = load_model('mf_model.h5', custom_objects={'rmse': rmse})
    model.summary()
   
    print('============================================================')
    print('Predict')
    result = model.predict([userID, movieID, userGender, userAge, userOccu, movieGenre])
   
    print('============================================================')
    print('Output Result')
    rating = np.clip(result, 1, 5).reshape(-1, 1)
    output = np.array( np.concatenate((ID, rating), axis=1))
    write_result(OUTPUT_FILE, output)


if __name__ == '__main__':
    main()

