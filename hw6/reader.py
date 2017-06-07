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

    print('movies:', np.array(movies).shape)
    return movies, all_genres


def read_user(filename):

    genders, ages, occupations = [[]]*6041, [[]]*6041, [ [0]*21 ]*6041
    categories = 21
    with open(filename, 'r', encoding='latin-1') as f:
        f.readline()
        for line in f:
            userID, gender, age, occu, zipcode = line[:-1].split('::')
            genders[int(userID)] = 0 if gender is 'F' else 1
            ages[int(userID)] = int(age)
            occupations[int(userID)] = to_categorical(int(occu), categories)
    
    print('genders:', np.array(genders).shape)
    print('ages:', np.array(ages).shape)
    print('occupations:', np.array(occupations).shape)
    return genders, ages, occupations


def read_train(filename):
    data = []
    with open(filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            dataID, userID, movieID, rating = row
            data.append( [int(dataID), int(userID), int(movieID), int(rating)] )

    print('Train data len:', len(data))
    return np.array(data)


def read_test(filename):
    data = []
    with open(filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            dataID, userID, movieID = row
            data.append( [dataID, int(userID), int(movieID)] )

    print('Test data len:', len(data))
    return np.array(data)


def preprocess(data, genders, ages, occupations, movies):

    if data.shape[1] == 4:
        print('Shuffle Data')
        np.random.seed(1019)
        index = np.random.permutation(len(data))
        data = data[index]

    print('Get ID')
    userID = np.array(data[:, 1], dtype=int)
    movieID = np.array(data[:, 2], dtype=int)

    print('Get Features')
    userGender = np.array(genders)[userID]
    userAge = np.array(ages)[userID]
    userOccu = np.array(occupations)[userID]
    movieGenre = np.array(movies)[movieID]

    print('Normalize Ages')
    std = np.std(userAge)
    userAge = userAge / std

    Rating = []
    if data.shape[1] == 4:
        print('Get Ratings')
        Rating = data[:, 3].reshape(-1, 1)

    print('userID:', userID.shape)
    print('movieID:', movieID.shape)
    print('userGender:', userGender.shape)
    print('userAge:', userAge.shape)
    print('userOccu:', userOccu.shape)
    print('movieGenre:', movieGenre.shape)
    print('Y:', np.array(Rating).shape)
    return userID, movieID, userGender, userAge, userOccu, movieGenre, Rating

def find_avg_Y(data):
    
    ratingSum = [0] * 6041
    ratingCount = [0] * 6041
    userID = data[:, 1]
    ratings = data[:, 3]
    for i, (uid, r) in enumerate(zip(userID, ratings)):
        print('\ri:', i, end='', flush=True)
        ratingSum[uid] += r
        ratingCount[uid] += 1

    ratingMean = [0] * 6041
    for i, (s, c) in enumerate(zip(ratingSum, ratingCount)):
        if c != 0:
            ratingMean[i] = s / c

    userAvgY = np.array(ratingMean)[userID]
    print('\ruserAvgY:', userAvgY.shape)
    return ratingMean
