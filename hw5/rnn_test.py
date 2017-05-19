# ML 2017 hw5
# Recurrent Neural Network (test)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import csv
from sys import argv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


def read_test(filename):
	
	test_text = []
	
	with open(filename, 'rb') as f:
		
		for line in f:
			idx, *text = str(line).split(',')
			if idx[2:] == 'id': continue
			test_text.append(''.join(text)[:-3])

	return test_text


def output_result(filename, result, categories):

	string_result = []

	for i, r in enumerate(result):
		select_index = np.where(r >= 0.15)[0]
		select_categories = categories[select_index]
		string = ' '.join(select_categories)
		string_result.append([i, string])
	
	with open(filename, 'w') as f:	
		w = csv.writer(f)
		w.writerow(['id', 'tags'])
		w.writerows(string_result)


MAX_SEQUENCE_LEN = 200
TOP_WORDS = 100000

# argv: [1]test_data.csv [2]prediction.csv [3]model.h5
def main():

	print('Read test data...')
	test_text = read_test(argv[1])
	categories = np.load('categories.npy')

	print('Tokenizer')
	tokenizer = Tokenizer(num_words=TOP_WORDS)
	texts = np.load('texts.npy')
	tokenizer.fit_on_texts(texts)	
	test_sequences = tokenizer.texts_to_sequences(test_text)
	test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LEN)
	print('Shape of test data:', test_data.shape)
	
	print('Load model')
	model = load_model(argv[3])
	model.summary()

	print('Predict')
	result = model.predict(test_data, verbose=1)

	print('Output Result')
	#output_result(argv[2], result, categories)
	output_result(argv[3][:-2] + 'csv', result, categories)

if __name__ == '__main__':
	main()
