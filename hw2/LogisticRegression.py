# ML hw2
# Logistic Regression

import numpy as np
import csv
import math
from sys import argv
from numpy.linalg import *

NUM_FEATURE = 106
OTHER_FEATURE = 0

ITERATION = 200
ETA = 1e-4
ADAGRAD = 0

NORMALIZE = 1
REGULARIZE = 0
LAMBDA = 0.01

np.set_printoptions(precision = 3, suppress = True)
np.seterr(divide='ignore', invalid='ignore')

def read_X(n):

	data = []
	with open(argv[n], "r", encoding="big5") as f:
		for line in list(csv.reader(f))[1:]:
			data.append([float(x) for x in line])

	return np.array(data), len(data)

def read_Y(n):

	data = []
	with open(argv[n], "r", encoding="big5") as f:
		for line in list(csv.reader(f)):
			data.append([float(x) for x in line])

	return np.array(data), len(data)

def alter_feature(X):
	global NUM_FEATURE
	X1 = X[:, [0]] * X[:, [0]]
	X2 = X[:, [0]] * X[:, [5]]
	NUM_FEATURE += 2
	index = list(range(106, NUM_FEATURE))
	return np.concatenate((X, X1, X2), axis = 1), index

def sigmoid(z):
	if not NORMALIZE: # z fix
		z = z * 1e-9 + 1
	return 1. / (1. + np.exp(-z))

def main():

	X_train, X_len = read_X(3)
	Y_train, Y_len = read_Y(4)
	print("X length:", X_len)
	print("Y length:", Y_len)

	# alter features
	other_index = []
	if OTHER_FEATURE:
		X_train, other_index = alter_feature(X_train)

	w = np.array([1e-4 for x in range(NUM_FEATURE)])
	b = 0.0

	# train normalization
	norm_index = [0, 1, 3, 4, 5] + other_index
	if NORMALIZE:
		np.random.seed(100)
		w = 0.001 * (np.random.random(NUM_FEATURE) * 2 - 1)
		b = 0.001 * (np.random.random() * 2 - 1)
		for i in norm_index:
			mu = np.mean(X_train[:, i], axis = 0)
			sig = np.std(X_train[:, i], axis = 0)
			X_train[:, i] = (X_train[:, i] - mu) / sig

	# train
	X = X_train
	Y = Y_train
	size = float(X_len)
	gw_all = np.zeros(w.shape)
	gb_all = 0.0
	for i in range(ITERATION):	
		# Ein
		correct = 0.0
		for j in range(X_len):
			z = np.dot(X[j], w) + b
			predict = 1.0 if z >= 0.0 else 0.0
			correct += 1 if predict == Y[j]	else 0
		print(i, "Correct Rate:", correct / size)

		gw = np.zeros(w.shape)
		gb = 0.0
		for j in range(X_len):	
			z = (np.dot(X[j], w) + b)
			gw += (sigmoid(z) - Y[j]) * X[j]
			gb += (sigmoid(z) - Y[j])

		if ADAGRAD:
			gw_all += (gw/size) ** 2
			gb_all += (gb/size) ** 2
			w_ = ETA * gw / np.sqrt(gw_all)
			b_ = ETA * gb / np.sqrt(gb_all)
		else:
			w_ = ETA * gw
			b_ = ETA * gb

		w -= w_
		b -= b_
		
	# test normalization
	X_test, X_test_len = read_X(5)
	if OTHER_FEATURE:
		X_test, other_index = alter_feature(X_test)
	for i in norm_index:
		mu = np.mean(X_test[:, i], axis = 0)
		sig = np.std(X_test[:, i], axis = 0)
		X_test[:, i] = (X_test[:, i] - mu) / sig

	# test
	result = []
	for i in range(X_test_len):
		z = np.dot(w, X_test[i]) + b
		p = sigmoid(z)
		if p >= 0.5:
			result += [1]
		else:
			result += [0]

	output(6, result)

def output(n, result):
	with open(argv[n], "w") as f:
		f.write("id,label\n")
		for i in range(len(result)):
			f.write(repr(i + 1) + "," + repr(result[i]) + "\n")
	
if __name__ == "__main__":
	main()
