# ML 2017 hw6
# Reader

import numpy as np
import csv

def to_categorical(index, categories):
    categorical = np.zeros(categories, dtype=int)
    categorical[index] = 1
    return list(categorical)


def read_movie(filename):

    def genre_to_number(genres, all_genres):
        result = []
        for g in genres.split('|'):
            if g not in all_genres:
                all_genres.append(g)
            result.append( all_genres.index(g) )
        return result, all_genres


    movies, all_genres = [[]] * 3953, []
    with open(filename, 'r', encoding='latin-1') as f:
        f.readline()
        for line in f:
            movieID, title, genre = line[:-1].split('::')
            genre_numbers, all_genres = genre_to_number(genre, all_genres)
            movies[int(movieID)] = genre_numbers
    
    categories = len(all_genres)
    for i, m in enumerate(movies):
        movies[i] = to_categorical(m, categories)

    return movies, all_genres, len(movies)


def read_user(filename):

    genders, ages, occupations = [[]]*6041, [[]]*6041, [[]]*6041
    categories = 21
    with open(filename, 'r', encoding='latin-1') as f:
        f.readline()
        for line in f:
            userID, gender, age, occu, zipcode = line[:-1].split('::')
            genders[int(userID)] = 0 if gender is 'F' else 1
            ages[int(userID)] = int(age)
            occupations[int(userID)] = to_categorical(int(occu), categories)
    
    return genders, ages, occupations, len(genders)


def read_train(filename):
    data = []
    with open(filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            dataID, userID, movieID, rating = row
            data.append( [int(dataID), int(userID), int(movieID), float(rating)] )
    return np.array(data)


def read_test(filename):
    data = []
    with open(filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            dataID, userID, movieID = row
            data.append( [int(dataID), int(userID), int(movieID)] )
    return np.array(data)


def preprocess(mode, data, genders, ages, movies):

    if mode == 'train':
        print('Shuffle Data')
        np.random.seed(2048)
        index = np.random.permutation(len(data))
        data = data[index]

    print('Get User/Movie ID')
    user_id = np.array(data[:, 1], dtype=int)
    movie_id = np.array(data[:, 2], dtype=int)
    
    print('Get User/Movie Features')
    user_genders = np.array(genders)[user_id].reshape(-1, 1)
    user_ages = np.array(ages)[user_id].reshape(-1, 1)
    movie_genres = np.array(movies)[movie_id]
    user_id = user_id.reshape(-1, 1)
    movie_id = movie_id.reshape(-1, 1)

    print('Normalize Ages')
    age_mean = np.mean(user_ages)
    age_std = np.std(user_ages)
    user_ages = (user_ages - age_mean) / age_std

    Y_rating = []
    if mode == 'train':
        print('Get Rating')
        Y_rating = data[:, 3].reshape(-1, 1)

    return user_id, movie_id, user_genders, user_ages, movie_genres, Y_rating
