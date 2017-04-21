# ML 2017 hw3 Train CNN

import numpy as np
import csv
from sys import argv
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

np.set_printoptions(precision = 6, suppress = True)

SHAPE = 48
CATEGORY = 7

BOX = 5
BOX_NUM = 25
POOLING = 2
DENSE1 = 100
DENSE2 = 7

BATCH = 100
EPOCHS = 10

AUGMENT = 2

def read_file(filename):

	X, Y = [], []
	with open(filename, "r", encoding="big5") as f:

		for line in list(csv.reader(f))[1:]:
			Y.append( float(line[0]) )
			X.append( [float(x) for x in line[1].split()] )

	return np.array(X), np_utils.to_categorical(Y, 7), len(X), len(Y)

def main():
	
	print("read data...")
	X, Y, X_len, Y_len = read_file(argv[1])
	X = X/255

	print("reshape data...")
	X = X.reshape(X.shape[0], SHAPE, SHAPE, 1)
	X_val = X[20000:]
	Y_val = Y[20000:]
	X = X[:20000]
	Y = Y[:20000]

	print("construct model...")
	S = 48
	model = Sequential()
	model.add(Conv2D(128, (5, 5), input_shape = (S, S, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (5, 5), input_shape = (S, S, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(32, (5, 5), input_shape = (S, S, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(units = 100, activation='relu'))
	model.add(Dense(units = 7, activation='softmax'))
	model.summary()

	print("compile model...")
	model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

	if AUGMENT == 1: 
		print("train with augmented data...")
		datagen = ImageDataGenerator(vertical_flip = False, horizontal_flip = True)
		datagen.fit(X)
		model.fit_generator(datagen.flow(X, Y, batch_size=32), samples_per_epoch=len(X), epochs=5, verbose=1, \
												validation_data=(X[:100], Y[:100]))
	elif AUGMENT == 2:
		print("train with self-augmented data...")
		X_flip = np.flip(X, 2)
		X_all = np.concatenate((X, X_flip), 0)
		Y_all = np.concatenate((Y, Y), 0)
		model.fit(X_all, Y_all, batch_size=BATCH, epochs=EPOCHS, verbose=1, validation_split=0.1)
	else:
		print("train with raw data...")
		model.fit(X, Y, batch_size=BATCH, epochs=EPOCHS, verbose=1, validation_split=0.1)
	
	print("evaluate train (val)...")
	score = model.evaluate(X_val, Y_val)

	print("train accuracy = " + repr(score[1]))

	print("save model...")
	model.save("{:.6f}".format(round(score[1], 6)) + ".h5")


if __name__ == '__main__':
	main()
