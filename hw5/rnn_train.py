# ML 2017 hw5
# Recurrent Neural Network (train)

import os
import numpy as np
import csv
from sys import argv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, MaxPooling1D
from keras.layers import LSTM, Dense
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
np.set_printoptions(precision=4, suppress=True)


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
	
	with open(filename, 'r') as f:
		
		for line in f:
			idx, label, *text = line.split(',')
			if idx == 'id': continue
			categories, label = find_labels(categories, label[1:-1])
			train_text.append(''.join(text))
			train_label.append(label)

	for i, label in enumerate(train_label):
		binary_label = np.zeros(len(categories))
		for c in label:
			binary_label[c] = 1.
		train_label[i] = list(binary_label)
	
	return train_text, train_label, categories


def read_test(filename):
	
	test_text = []
	
	with open(filename, 'r') as f:
		
		for line in f:
			idx, *text = line.split(',')
			if idx == 'id': continue
			test_text.append(''.join(text))

	return test_text


def read_word_vector(filename):
	
	embedding_index = {}

	with open(filename, 'r') as f:
		
		for line in f:
			values = line.split()
			word = values[0]
			vector = np.array(values[1:], dtype='float32')
			embedding_index[word] = vector

	return embedding_index


MAX_SEQUENCE_LEN = 150
TOP_WORDS = 20000
EMBEDDING_DIM = 100

# argv: [1]train_data.csv [2]test_data.csv
def main():
	
	print('Read train data...')
	train_text, train_label, categories = read_train(argv[1])
	
	print('Read test data...')
	test_text = read_test(argv[2])

	print('Tokenizer')
	tokenizer = Tokenizer(num_words=TOP_WORDS)
	tokenizer.fit_on_texts(train_text + test_text)	
	
	train_sequences = tokenizer.texts_to_sequences(train_text)
	test_sequences = tokenizer.texts_to_sequences(test_text)
	
	train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LEN)
	test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LEN)
	train_label = np.array(train_label)

	print('Shape of train data:', train_data.shape)
	print('Shape of train label:', train_label.shape)
	print('Shape of test data:', test_data.shape)
	
	print('Embedding Layer')
	word_index = tokenizer.word_index
	embedding_index = read_word_vector('glove.6b.100d.txt')	
	num_words = min(TOP_WORDS, len(word_index))
	embedding_matrix = np.zeros([num_words, EMBEDDING_DIM])
	if '--load-embedding' in argv:
		embedding_matrix = np.load('embedding_matrix.npy')
	else:
		for word, i in word_index.items():
			if i >= TOP_WORDS: continue
			embedding_vector = embedding_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
	
	embedding_layer = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LEN, trainable=False)

	print('Construct model')
	model = Sequential()
	model.add(embedding_layer)
	model.add(LSTM(MAX_SEQUENCE_LEN))
	model.add(Dense(38, activation='sigmoid'))
	model.summary()

	print('Compile Model')
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	print('Train Model')
	EPOCHS = 5
	history = model.fit(train_data, train_label, batch_size=128, epochs=EPOCHS)
	h = history.history

	print('Evaluate ', end='')
	score = model.evaluate(train_data, train_label, verbose=1)
	print('acc = %f' % score[1])

	print('Save')
	model.save('{:.6f}'.format(score[1]) + '_' + repr(EPOCHS) + 'e.h5')
	
	np.save('{:.6}'.format(score[1]) + '_' + repr(EPOCHS) + 'e_history.npy', h['acc'])
	np.save('categories.npy', categories)
	np.save('texts.npy', train_text + test_text)
	
	if '--load-embedding' not in argv:
		np.save('embedding_matrix.npy', embedding_matrix)


if __name__ == '__main__':
	main()
