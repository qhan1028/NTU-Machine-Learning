# ML 2017 hw5
# Recurrent Neural Network (train)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import csv
from sys import argv
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, MaxPooling1D
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam


def find_labels(categories, y_string):
	
	y_labels = []

	for y in y_string.split():

		if y not in categories:
			categories.append(y)
		
		y_labels.append( categories.index(y) )
	
	return categories, y_labels


def read_train(filename):

	categories = []
	train_text = []
	train_label = []
	
	with open(filename, 'rb') as f:
		
		for line in f:
			idx, label, *text = str(line).split(',')
			if idx[2:] == 'id': continue
			categories, label = find_labels(categories, label[1:-1])
			train_text.append(''.join(text)[:-3])
			train_label.append(label)

	for i, label in enumerate(train_label):
		binary_label = np.zeros(len(categories))
		for c in label:
			binary_label[c] = 1.
		train_label[i] = list(binary_label)
	
	return train_text, train_label, categories


def read_test(filename):
	
	test_text = []
	
	with open(filename, 'rb') as f:
		
		for line in f:
			idx, *text = str(line).split(',')
			if idx[2:] == 'id': continue
			test_text.append(''.join(text)[:-3])

	return test_text


def read_word_vector(filename):
	
	embedding_index = {}

	with open(filename, 'rb') as f:
		
		i = 0
		for line in f:
			print('\r%d' % i, end='', flush=True)
			values = str(line).split()
			word = values[0][2:]
			vector = np.array(values[1:-1] + [values[-1][:-3]], dtype='float32')
			embedding_index[word] = vector
			i += 1
		print('')

	return embedding_index


def split_validation(X, Y, split_ratio):
	indices = np.arange(X.shape[0])  
	np.random.shuffle(indices) 
	
	X_data = X[indices]
	Y_data = Y[indices]
	
	num_validation_sample = int(split_ratio * X_data.shape[0] )
	
	X_train = X_data[num_validation_sample:]
	Y_train = Y_data[num_validation_sample:]

	X_val = X_data[:num_validation_sample]
	Y_val = Y_data[:num_validation_sample]

	return (X_train,Y_train),(X_val,Y_val)


def f1_score(y_true,y_pred):
	thresh = 0.4
	y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
	tp = K.sum(y_true * y_pred)
	
	precision=tp/(K.sum(y_pred))
	recall=tp/(K.sum(y_true))
	return 2*((precision*recall)/(precision+recall))


EMBEDDING_DIM = 100

# argv: [1]train_data.csv [2]test_data.csv
def main():

	print('==================================================================')	
	print('Read train data')
	train_text, train_label, categories = read_train(argv[1])
	print('Read test data')
	test_text = read_test(argv[2])

	print('==================================================================')	
	print('Tokenizer')
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(train_text + test_text)	
	print('Convert to index sequences')	
	train_sequences = tokenizer.texts_to_sequences(train_text)
	test_sequences = tokenizer.texts_to_sequences(test_text)
	print('Pad sequences')
	train_data = pad_sequences(train_sequences)
	train_label = np.array(train_label)
	MAX_SEQUENCE_LEN = train_data.shape[1]
	test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LEN)

	print('==================================================================')	
	print('Split validation')
	(X_train, Y_train), (X_val, Y_val) = split_validation(train_data, train_label, 0.1)
	print('Shape of X:', train_data.shape)
	print('Shape of Y:', train_label.shape)
	print('Shape of X_train:', X_train.shape)
	print('Shape of Y_train:', Y_train.shape)
	print('Shape of X_val:', X_val.shape)
	print('Shape of Y_val:', Y_val.shape)
	print('Shape of test data:', test_data.shape)
	
	print('==================================================================')	
	print('Embedding Layer')
	word_index = tokenizer.word_index
	num_words = len(word_index)
	embedding_matrix = np.zeros([num_words, EMBEDDING_DIM])
	if '--load-embedding' in argv:
		embedding_matrix = np.load('embedding_matrix.npy')
	else:
		embedding_dict = read_word_vector('glove.6b.100d.txt')	
		for word, i in word_index.items():
			if i < num_words:
				embedding_vector = embedding_dict.get(word)
				if embedding_vector is not None:
					embedding_matrix[i] = embedding_vector
	
	embedding_layer = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LEN, trainable=False)

	print('==================================================================')	
	print('Construct model')
	model = Sequential()
	model.add(embedding_layer)
	model.add(GRU(128, activation='tanh'))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(38, activation='sigmoid'))
	model.summary()

	print('Compile Model')
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[f1_score])

	print('Train Model')
	EPOCHS = 100
	history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=128, epochs=EPOCHS)
	h = history.history

	score = h['val_f1_score'][-1]
	print('last val f1_score = %f' % score)

	print('==================================================================')	
	print('Save')
	model.save('{:.6f}'.format(score) + '_' + repr(EPOCHS) + 'e.h5')
	
	np.savez('{:.6}'.format(score) + '_' + repr(EPOCHS) + 'e_history.npz', h['f1_score'], h['val_f1_score'])
	np.save('categories.npy', categories)
	np.save('texts.npy', train_text + test_text)
	
	if '--load-embedding' not in argv:
		np.save('embedding_matrix.npy', embedding_matrix)


if __name__ == '__main__':
	main()
