# ML 2017 hw3 Test by Vote

import numpy as np
import csv
from sys import argv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from os.path import isfile
from keras.models import load_model

np.set_printoptions(precision = 6, suppress = True)

CATEGORY = 7

READ_FROM_NPZ = 1
SHAPE = 48

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

def read_model_list(filename):

	model_list = []
	with open(filename, "r") as f:
		for line in f:
			weight = float(line.split()[0])
			model_name = line.split()[1]
			model_list.append((weight, model_name))

	return model_list

def write_file(filename, result):

	with open(filename, "w", encoding="big5") as f:	
		f.write("id,label\n")
		for i in range(len(result)):
			predict = np.argmax(result[i])
			f.write(repr(i) + "," + repr(predict) + "\n")

# argv: [1]test.csv [2]predict.csv [3]model_list.txt
def main():
	
	X_test = []
	if READ_FROM_NPZ:
		print("read from npz...")
		data = np.load("data.npz")
		X_test = data['arr_2']
	else:
		print("read test data...")
		X_test = read_test(argv[1])

	print("reshape data...")
	X_test = X_test/255
	X_test = X_test.reshape(X_test.shape[0], SHAPE, SHAPE, 1)

	print("load model list...")
	model_list = read_model_list(argv[3])
	nb_models = len(model_list)
	print("total models: " + repr(nb_models))

	print("predict...")
	result = np.zeros([len(X_test), CATEGORY])
	for i in range(nb_models):
		weight = model_list[i][0]
		model_name = model_list[i][1]
		print("model: " + model_name + " weight: " + repr(weight), flush=True)
		predict = np.zeros(result.shape)
		if isfile(model_name + ".npy"):
			predict = np.load(model_name + ".npy")
		else:
			model = load_model(model_name)
			predict = model.predict(X_test, batch_size = 128, verbose = 1)
			np.save(model_name + ".npy", predict)
		result += predict * weight
		print("", flush=True)

	print("output result...")
	write_file(argv[2], result)

if __name__ == "__main__":
	main()
