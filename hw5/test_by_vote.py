# ML 2017 hw5
# Recurrent Neural Network (test by vote)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import csv
from sys import argv
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


def read_test(filename):
    
    test_text = []
    
    with open(filename, 'r', encoding='UTF-8') as f:
        
        for line in f:
            idx, *text = str(line).split(',')
            if idx == 'id': continue
            test_text.append(''.join(text))

    return test_text

def read_model_list(filename):

    model_list = []
    with open(filename, "r") as f:
        for line in f:
            weight = float(line.split()[0])
            model_name = line.split()[1]
            model_list.append((weight, model_name))

    return model_list

def output_result(filename, result, categories, threshold):

    string_result = []

    for i, r in enumerate(result):
        select_index = np.where(r >= threshold)[0]
        if len(select_index) == 0:
            select_index = [np.argmax(r)]
        select_categories = categories[select_index]
        string = ' '.join(select_categories)
        string_result.append([i, string])
    
    with open(filename, 'w') as f:    
        w = csv.writer(f)
        w.writerow(['id', 'tags'])
        w.writerows(string_result)

def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred)
    
    precision=tp/(K.sum(y_pred))
    recall=tp/(K.sum(y_true))
    return 2*((precision*recall)/(precision+recall))


MAX_SEQUENCE_LEN = 306
THRESHOLD = 0.3

# argv: [1]test_data.csv [2]prediction.csv
def main():

    print('==================================================================')    
    print('Read test data and categories.')
    test_text = read_test(argv[1])
    categories = np.load('categories.npy')

    print('==================================================================')    
    print('Predict')
    model_list = read_model_list('model_list.txt')
    nb_models = len(model_list)
    print('Total models: %d' % nb_models)

    sum_result = np.zeros([len(test_text), len(categories)])
    sum_weight = 0
    for (weight, name) in model_list:

        print('model: ' + name + ', weight: %f' % weight)
        tokenizer = Tokenizer()
        tokenizer.word_index = np.load(name[:-3] + '_word_index.npy').item()
        test_sequences = tokenizer.texts_to_sequences(test_text)
        test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LEN)
    
        model = load_model(name, custom_objects={'f1_score': f1_score})
        sum_result += model.predict(test_data, verbose=0)
        sum_weight += weight
    
    sum_result /= sum_weight

    print('==================================================================')    
    print('Output result. threshold: %f' % THRESHOLD)
    output_result(argv[2], sum_result, categories, THRESHOLD)

if __name__ == '__main__':
    main()
