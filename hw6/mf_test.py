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
    movies, all_genres, n_movies = read_movie(DATA_DIR + '/movies.csv')
    genders, ages, occupations, n_users = read_user(DATA_DIR + '/users.csv')
    test = read_test(DATA_DIR + '/test.csv')
    ID = np.array(test[:, 0], dtype=int).reshape(-1, 1)
    print('n_movies:', n_movies, ', n_users:', n_users)
    print('Test data len:', len(test))

    print('============================================================')
    print('Preprocess Data')
    user_id, movie_id, user_genders, user_ages, movie_genres, _Y = \
        preprocess('test', test, genders, ages, movies)

    print('============================================================')
    print('Load Model')

    def rmse(y_true, y_pred):
        mse = K.mean((y_pred - y_true) ** 2)
        return K.sqrt(mse)

    model = load_model('mf_model.h5', custom_objects={'rmse': rmse})
    model.summary()
   
    print('============================================================')
    print('Predict')
    result = model.predict([user_id, movie_id, user_genders, user_ages, movie_genres])
   
    print('============================================================')
    print('Output Result')
    rating = np.round( np.clip(result, 1, 5) ).reshape(-1, 1)
    output = np.array( np.concatenate((ID, rating), axis=1), dtype=int)
    print(output)
    write_result(OUTPUT_FILE, output)


if __name__ == '__main__':
    main()

