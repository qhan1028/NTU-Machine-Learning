# ML 2017 hw6
# Plot Model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import load_model
from sklearn.manifold import TSNE
from reader import *

DATA_DIR = './data'
MODEL = sys.argv[1]

if '--tsne' in sys.argv:
    
    print('============================================================')
    print('Read Data')
    movies, all_genres = read_movie(DATA_DIR + '/movies.csv')
    print('movies:', np.array(movies).shape)

    print('Get Movie List')
    movie_list = []
    for i, m in enumerate(movies):
        if np.sum(m) != 0:
            movie_list.append(i)
    print('movie list:', np.array(movie_list).shape)

    print('============================================================')
    print('Load Model')
    def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )
    model = load_model(MODEL, custom_objects={'rmse': rmse})

    get_emb_movieID = K.function([model.get_layer('in_movieID').input], \
                                 [model.get_layer('vec_movieID').output])

    movie_list_reshape = np.array(movie_list).reshape(-1, 1)
    movie_512d = get_emb_movieID([movie_list_reshape])[0]
    print('movie_512d:', movie_512d.shape)
    np.save('movie_512d.npy', movie_512d)

    print('============================================================')
    print('T-SNE')
    movie_2d = []
    if os.path.exists('movie_2d.npy'):
        print('load 2d from file.')
        movie_2d = np.load('movie_2d.npy')
    else:
        print('can\'t use TNSE.')
        tsne = TSNE(n_components=2)
        movie_2d = tsne.fit_transform(X=movie_512d[:10])
    print('movie_2d:', movie_2d.shape)

    print('============================================================')
    print('Plot 2d Graph')
    
    categories = []
    category_index = [[0, 1, 2, 3], [5, 6, 12, 14], [4, 7, 11, 13, 17], [8, 9, 10, 16]]
    colors = ['red', 'green', 'blue', 'black', 'lightgray']

    for i in range(4):
        print(colors[i] + ':', np.array(all_genres)[category_index[i]] )
    print('lightgray: [\'other\']')
    
    for idx in category_index:
        category_array = np.zeros(18)
        category_array[idx] = 1
        categories.append( category_array )

    def find_category(in_c):
        max_similar, color = 0, 4
        for i, c in enumerate(categories):
            similar = np.sum(c * in_c)
            if similar > max_similar:
                max_similar, color = similar, i
        return colors[color]

    plt.clf()
    for (ID, point) in zip(movie_list, movie_2d):
        print('\rmovie ID: %d' % ID, end='', flush=True)
        genre = movies[ID]
        plt.plot(point[0], point[1], '.', color=find_category(genre))
    print('')
    
    plt.savefig(MODEL[:-3] + '_emb.png', dpi=300)
    plt.show()


if '-h' in sys.argv:

    print('============================================================')
    print('Plot History')
    HIS_FILE = MODEL[:-3] + '_his.npz'
    history = np.load(HIS_FILE)
    rmse = history['rmse']
    val_rmse = history['val_rmse']
    
    plt.clf()
    plt.plot(rmse, 'b')
    plt.plot(val_rmse, 'r')

    plt.legend(['RMSE', 'val RMSE'], loc='upper right')
    plt.xlabel('epochs')
    plt.ylabel('RMSE')
    plt.title(MODEL)
    plt.savefig(MODEL[:-3] + '_his.png', dpi=300)
    plt.show()
