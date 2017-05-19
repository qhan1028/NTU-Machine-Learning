# ML 2017 hw3 Train CNN

import numpy as np
import csv
from sys import argv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

np.set_printoptions(precision = 6, suppress = True)

SHAPE = 48
CATEGORY = 7

READ_FROM_NPZ = 1
AUGMENT = 0
SAVE_HISTORY = 1

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

# argv: [1]train.csv
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
	X = X.reshape(X.shape[0], 48*48)

	print("construct model...")
	model = Sequential()
	model.add(Dense(units = 512, activation='relu', input_dim=48*48))
	model.add(Dense(units = 512, activation='relu'))
	model.add(Dense(units = 512, activation='relu'))
	model.add(Dense(units = 256, activation='relu'))
	model.add(Dense(units = 7, activation='softmax'))
	model.summary()

	print("compile model...")
	model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

	VAL = 2400
	BATCH = 128
	EPOCHS = 30
	print("train...")
	earlyStopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')
	history = model.fit(X, Y, batch_size=BATCH, epochs=EPOCHS, verbose=1, validation_split=0.1)
	h = history.history
	val_acc = h['val_acc'][-1]
	print("train accuracy (val last) = " + "{:6f}".format(val_acc))
	if SAVE_HISTORY:
		print("save history...")
		np.savez("dnn_" + "{:.6f}".format(val_acc) + "_history.npz", h['acc'], h['val_acc'])
		
	print("save model...")
	model.save("dnn_" + "{:.6f}".format(val_acc) + ".h5")

if __name__ == '__main__':
	main()
