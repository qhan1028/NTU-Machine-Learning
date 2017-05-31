# ML 2017 hw6
# reader

import numpy as np

def read_movie(filename):

    def genre_to_number(genres, all_genres):
        result = []
        for g in genres.split('|'):
            if g not in all_genres:
                all_genres.append(g)
            result.append( all_genres.index(g) )
        return result, all_genres

    movies, all_genres = [], []
    with open(filename, 'r', encoding='latin-1') as f:
        f.readline()
        for line in f:
            movieID, title, genre = line[:-1].split('::')
            genre_numbers, all_genres = genre_to_number(genre, all_genres)
            movies.append( (int(movieID), genre_numbers) )
    return movies, all_genres, len(movies)


def read_user(filename):

    users = []
    with open(filename, 'r', encoding='latin-1') as f:
        f.readline()
        for line in f:
            userID, gender, age, occupation, zipcode = line[:-1].split('::')
            gender = 0 if gender is 'F' else 1
            users.append( (int(userID), gender, int(age), int(occupation), zipcode) )
    return users, len(users)


import csv

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
