# ML 2017 hw3 Train CNN

import numpy as np
import csv
from sys import argv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

np.set_printoptions(precision = 6, suppress = True)

SHAPE = 48
CATEGORY = 7

READ_FROM_NPZ = 1
AUGMENT = 1

SEMI_THRES = 0.5

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

def semi_data(X_test, result):
	
	X, Y = [], []
	for i in range(len(result)):
		p = np.max(result[i])
		idx = np.argmax(result[i])
		if p >= SEMI_THRES:
			X.append(X_test[i])
			Y.append(idx)
	
	return np.array(X), np_utils.to_categorical(Y, CATEGORY)

# argv: [1]train.csv [2]test.csv [3]old_model.h5 [4]semi_model.h5 [5]start_epoch [6]end_epoch
def main():
	
	X_train, Y_train, X_test = [], [], []
	if READ_FROM_NPZ:
		print("read from npz...")
		data = np.load("data.npz")
		X_train = data['arr_0']
		Y_train = data['arr_1']
		X_test = data['arr_2']
	else:
		print("read train data...")
		X_train, Y_train = read_train(argv[1])
		print("read test data...")
		X_test = read_test(argv[2])

	print("reshape data...")
	X_train = X_train/255
	X_train = X_train.reshape(X_train.shape[0], SHAPE, SHAPE, 1)
	X_test = X_test/255
	X_test = X_test.reshape(X_test.shape[0], SHAPE, SHAPE, 1)

	print("load original model...")
	old_model = load_model(argv[3])

	print("predict test data...")
	result = old_model.predict(X_test, batch_size=128, verbose=1)
	X_semi, Y_semi = semi_data(X_test, result)

	X = np.concatenate((X_train, X_semi), 0)
	Y = np.concatenate((Y_train, Y_semi), 0)
	print("total X: " + repr(len(X)))

	print("load semi model...")
	semi_model = load_model(argv[4])

	start_epoch = int(argv[5])
	end_epoch = int(argv[6])

	VAL = 2400
	BATCH = 128
	EPOCHS = 100
	score = [0]
	if AUGMENT == 1: 
		print("train with augmented data...")
		datagen = ImageDataGenerator(vertical_flip=False, horizontal_flip=True, \
																 height_shift_range=0.1, width_shift_range=0.1)
		Xv = X[:VAL]
		Yv = Y[:VAL]
		datagen.fit(X[VAL:], seed=1028)
		history = semi_model.fit_generator(datagen.flow(X[VAL:], Y[VAL:], batch_size=BATCH, seed=1028), samples_per_epoch=len(X), \
																	epochs=end_epoch, verbose=1, validation_data=(Xv, Yv), initial_epoch=start_epoch)
		score.append(round(history.history['val_acc'][-1], 6))
		print("train accuracy (last val) = " + repr(score[1]))
	else:
		print("train with raw data...")
		earlyStopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')
		semi_model.fit(X, Y, batch_size=BATCH, epochs=EPOCHS, verbose=1, validation_split=0.1, callbacks=[earlyStopping])
		print("evaluate train...")
		score = semi_model.evaluate(X, Y)
		print("train accuracy (all) = " + repr(score[1]))

	print("save model...")
	semi_model.save("{:.6f}".format(round(score[1], 6)) + "_" + argv[4])


if __name__ == '__main__':
	main()
