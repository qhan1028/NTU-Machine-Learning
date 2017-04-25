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

# argv: 1: train.csv 2: model.h5 3: start_epoch 4: end_epoch
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

	print("reshape data...")
	X = X/255
	X = X.reshape(X.shape[0], SHAPE, SHAPE, 1)

	print("load model and next epoch...")
	model = load_model(argv[2])
	start_epoch = int(argv[3])
	end_epoch = int(argv[4])

	VAL = 2400
	BATCH = 100
	EPOCHS = 100
	score = [0]
	if AUGMENT == 1: 
		print("train with augmented data...")
		datagen = ImageDataGenerator(vertical_flip=False, horizontal_flip=True, \
																 height_shift_range=0.1, width_shift_range=0.1)
		Xv = X[:VAL]
		Yv = Y[:VAL]
		datagen.fit(X[VAL:], seed=1028)
		history = model.fit_generator(datagen.flow(X[VAL:], Y[VAL:], batch_size=BATCH, seed=1028), samples_per_epoch=len(X), \
																	epochs=end_epoch, verbose=1, validation_data=(Xv, Yv), initial_epoch=start_epoch)
		score.append(round(history.history['val_acc'][-1], 6))
		print("train accuracy (last val) = " + repr(score[1]))
	else:
		print("train with raw data...")
		earlyStopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')
		model.fit(X, Y, batch_size=BATCH, epochs=EPOCHS, verbose=1, validation_split=0.1, callbacks=[earlyStopping])
		print("evaluate train...")
		score = model.evaluate(X, Y)
		print("train accuracy (all) = " + repr(score[1]))

	print("save model...")
	model.save("{:.6f}".format(round(score[1], 6)) + "_" + argv[2])


if __name__ == '__main__':
	main()
