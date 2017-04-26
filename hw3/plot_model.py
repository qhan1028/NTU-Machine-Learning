# ML2017 hw3 Plot Model

import numpy as np
from sys import argv
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.models import load_model
from keras.utils import plot_model, np_utils
from sklearn.metrics import confusion_matrix

READ_FROM_NPZ = 1
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

# argv: [1]train.csv [2]model.h5
def main():

	X, Y = [], []
	if READ_FROM_NPZ:
		print("read from npz...")
		data = np.load("data.npz")
		X = data['arr_0']
		Y = data['arr_1']
	else:
		print("read train data...")
		X, Y = read_train(argv[1])

	model_name = argv[2]
	print("load model...")
	model = load_model(model_name)
	model.summary()
	print("save structure figure...")
	plot_model(model, show_layer_names=False, show_shapes=True, to_file=model_name[:-3] + ".png")

	print("plot confusion matrix...")
	X = X.reshape(X.shape[0], 48, 48, 1)
	Y = np.argmax(Y, 1)
	predict = model.predict(X, verbose=1, batch_size=128)
	Y_predict = np.argmax(predict, 1)
	cm = confusion_matrix(Y, Y_predict)
	print(Y)
	print(Y_predict)
	print(cm)

if __name__ == "__main__":
	main()
