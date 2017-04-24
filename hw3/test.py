# ML 2017 hw3 Test CNN

import numpy as np
import csv
from sys import argv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.models import load_model

np.set_printoptions(precision = 6, suppress = True)

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

def write_file(filename, result):

	with open(filename, "w", encoding="big5") as f:
		
		f.write("id,label\n")
		for i in range(len(result)):
			predict = np.argmax(result[i])
			f.write(repr(i) + "," + repr(predict) + "\n")

# argv: [1]test.csv [2]predict.csv [3]model.h5
def main():
	
	print("read test data...")
	data = read_test(argv[1])

	print("reshape test data...")
	data = data / 255
	data = data.reshape(data.shape[0], 48, 48, 1)

	print("load model...")
	model = load_model(argv[3])

	print("predict...")
	result = model.predict(data, batch_size = 128, verbose = 1)

	print("output result...")
	write_file(argv[2], result)

if __name__ == "__main__":
	main()
