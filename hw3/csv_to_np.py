# ML 2017 hw3 transform data from csv to numpy

import numpy as np
import csv
from sys import argv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.utils import np_utils

SHAPE = 48
CATEGORY = 7

def read_train(filename):

	X, Y = [], []
	with open(filename, "r", encoding="big5") as f:
		count = 0
		for line in list(csv.reader(f))[1:]:
			Y.append( float(line[0]) )
			X.append( [float(x) for x in line[1].split()] )
			count += 1
			print("\rX_train: " + repr(count), end="", flush=True)
		print("", flush=True)

	return np.array(X), np_utils.to_categorical(Y, CATEGORY)

def read_test(filename):

	data = []
	with open(filename, "r", encoding="big5") as f:
		count = 0
		for line in list(csv.reader(f))[1:]:
			data.append( [float(x) for x in line[1].split()] )
			count += 1
			print("\rX_test: " + repr(count), end="", flush=True)
		print("", flush=True)

	return np.array(data)

# argv: [1]train.csv [2]test.csv
if __name__ == '__main__':
	print("read train data...")
	X_train, Y_train = read_train(argv[1])
	print("read test data...")
	X_test = read_test(argv[2])
	print("save as npz...")
	np.savez("data.npz", X_train, Y_train, X_test)
