# ML 2017 hw3 Semi-Supervised Train CNN

import numpy as np
import csv
from sys import argv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.models import Sequential, load_model
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

# argv: [1]train.csv [2]test.csv [3]model.h5
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

	print("load model...")
	old_model = load_model(argv[3])

	print("predict test data...")
	result = old_model.predict(X_test, batch_size=128, verbose=1)
	X_semi, Y_semi = semi_data(X_test, result)

	X = np.concatenate((X_train, X_semi), 0)
	Y = np.concatenate((Y_train, Y_semi), 0)
	print("total X: " + repr(len(X)))

	print("construct new model...")
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(MaxPooling2D((2, 2)))

	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(MaxPooling2D((2, 2)))

	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	
	model.add(Dense(units = 256, activation='relu'))
	model.add(Dense(units = 128, activation='relu'))
	model.add(Dense(units = 64, activation='relu'))
	model.add(Dense(units = 7, activation='softmax'))
	model.summary()

	print("compile new model...")
	model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

	VAL = 2400
	BATCH = 128
	EPOCHS = 20
	score = [0]
	if AUGMENT == 1: 
		print("train with augmented data...")
		datagen = ImageDataGenerator(vertical_flip=False, horizontal_flip=True, \
																 height_shift_range=0.1, width_shift_range=0.1)
		Xv = X[:VAL]
		Yv = Y[:VAL]
		datagen.fit(X[VAL:], seed=1028)
		history = model.fit_generator(datagen.flow(X[VAL:], Y[VAL:], batch_size=BATCH, seed=1028), samples_per_epoch=len(X), \
																	epochs=EPOCHS, verbose=1, validation_data=(Xv, Yv))
		score.append(round(history.history['val_acc'][-1], 6))
		print("train accuracy (last val) = " + repr(score[1]))
	else:
		print("train with raw data...")
		earlyStopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')
		model.fit(X, Y, batch_size=BATCH, epochs=EPOCHS, verbose=1, validation_split=0.1, callbacks=[earlyStopping])
		print("evaluate train...")
		score = model.evaluate(X, Y)
		print("train accuracy (all) = " + repr(score[1]))

	print("save new model...")
	model.save("semi_" + "{:.6f}".format(round(score[1], 6)) + "_" + argv[3])


if __name__ == '__main__':
	main()
