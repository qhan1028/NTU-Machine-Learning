# ML hw2
# Probabilistic Generative Model

import numpy as np
import csv
import math
from sys import argv
from numpy.linalg import *

NUM_FEATURE = 106
THRESHOLD = 0

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

def F(C, x, mu, sig_inv, x_t, mut):
	
	result = C * np.exp( (-1./2.) * np.dot( np.dot(x - mu, sig_inv), x_t - mut) )
	return result

def main():
	
	X_train, X_len = read_X(3)
	Y_train, Y_len = read_Y(4)

	print("X length:", X_len)
	print("Y length:", Y_len)

	X = X_train
	Y = Y_train
	NUM_FEATURE = len(X[0])

	mu1 = np.array( [0.0 for x in range(NUM_FEATURE)] ) # mean
	mu2 = np.array( [0.0 for x in range(NUM_FEATURE)] ) # mean
	sig1 = np.array( [ [0.0 for x in range(NUM_FEATURE)] for _ in range(NUM_FEATURE) ] ) # cov
	sig2 = np.array( [ [0.0 for x in range(NUM_FEATURE)] for _ in range(NUM_FEATURE) ] ) # cov

	# mean
	mu1_count = 0
	for i in range(X_len):
		if Y[i]:
			mu1 += X[i]
			mu1_count += 1
		else:
			mu2 += X[i]

	mu2_count = X_len - mu1_count
	mu1 /= float(mu1_count)
	mu2 /= float(mu2_count)
	mu1_t = mu1.reshape(-1, 1)
	mu2_t = mu2.reshape(-1, 1)

	# covariance matrix
	for i in range(X_len):
		x = X[i]
		x_t = x.reshape(-1, 1)
		if Y[i]:
			sig1 += np.dot( (x_t - mu1_t), [x - mu1] )
		else:
			sig2 += np.dot( (x_t - mu2_t), [x - mu2] )
	
	sig1 /= float(mu1_count)
	sig2 /= float(mu2_count)
	p1 = float(mu1_count) / float(X_len)
	p2 = 1.0 - p1
	sig = p1 * sig1 + p2 * sig2 # use same sigma

	# determinant & inverse
	sig_det = det(sig)
	sig_inv = pinv(sig)
	C = 1. / ( pow(2. * math.pi, NUM_FEATURE / 2.) * pow(abs(sig_det), 1./2.) )

	# Ein
	correct = 0
	for i in range(X_len):
		x = X[i]
		x_t = X[i].reshape(-1, 1)
		f1 = F(C, x, mu1, sig_inv, x_t, mu1_t)
		f2 = F(C, x, mu2, sig_inv, x_t, mu2_t)

		P1 = (f1 * p1) / (f1 * p1 + f2 * p2)
		predict = 1 if P1 >= 0.5 else 0
		if predict == Y[i]:
			correct += 1

	curr = float(correct / X_len)
	print("Correct Rate:", curr)

	
	# test
	X_test, X_test_len = read_X(5)
	X = X_test
	result = []
	for i in range(X_test_len):
		x = X[i]
		x_t = X[i].reshape(-1, 1)
		f1 = F(C, x, mu1, sig_inv, x_t, mu1_t)
		f2 = F(C, x, mu2, sig_inv, x_t, mu2_t)
		P1 = (f1 * p1) / (f1 * p1 + f2 * p2)
		if P1 >= 0.5:
			result += [1]
		else:
			result += [0]

	# output result
	output(6, result)
	
def output(n, result):

	with open(argv[n], "w") as f:
		f.write("id,label\n")
		for i in range(len(result)):
			f.write(repr(i + 1) + "," + repr(result[i]) + "\n")

if __name__ == "__main__":
	main()
