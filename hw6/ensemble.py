# ML 2017 hw6 
# Ensemble

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import csv
import numpy as np
import keras.backend as K
from keras.models import Model, load_model
from reader import *

# argv: [1]data directory, [2]output file, [3]model_list.txt
DATA_DIR = sys.argv[1]
MODEL_DIR = './model'
OUTPUT_FILE = sys.argv[2]
MODEL_LIST = sys.argv[3]

def read_list(filename):
    weights, names = [], []
    with open(filename, 'r') as f:
        for line in f:
            row = line.split()
            weights.append(float(row[0]))
            names.append(row[1])

    return weights, names

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

    test = read_test(DATA_DIR + '/test.csv')
    ID = np.array(test[:, 0]).reshape(-1, 1)

    print('============================================================')
    print('Preprocess Data')
    userID, movieID, userGender, userAge, userOccu, movieGenre, _Y = \
        preprocess(test, genders, ages, occupations, movies)

    print('============================================================')
    print('Read Model List')
    weights, names = read_list(MODEL_LIST)

    print('============================================================')
    print('Predict')
    def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )

    sum_result = np.zeros((len(test), 1))
    sum_weights = 0
    for w, name in zip(weights, names):
        
        print('weight: %f' % w + ', name: ' + name, end=' ', flush=True)
        model = load_model(MODEL_DIR + '/' + name, custom_objects={'rmse': rmse})
        print('[loaded]', end=' ', flush=True)
        result = np.zeros(len(test))
        if name[:2] == 'mf':
            if name[3:9] == 'simple':
                result = model.predict([userID, movieID])
            else:
                result = model.predict([userID, movieID, userGender, userAge, userOccu, movieGenre])
        elif name[:3] == 'dnn':
            if name[4:10] == 'simple':
                result = model.predict([userID, movieID])
            else:
                result = model.predict([userID, movieID, userGender, userAge, userOccu, movieGenre])
        print('[predicted]', flush=True)

        sum_result += result * w
        sum_weights += w
    
    sum_result /= sum_weights

    print('============================================================')
    print('Output Result')
    rating = np.clip(sum_result, 1, 5).reshape(-1, 1)
    output = np.array( np.concatenate((ID, rating), axis=1))
    write_result(OUTPUT_FILE, output)

if __name__ == '__main__':
    main()
