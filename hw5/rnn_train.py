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
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
    count = np.zeros(38)
    
    with open(filename, 'r', encoding='UTF-8') as f:
    
        f.readline()
        for line in f:
            start = line.find('\"')
            end = line.find('\"', start+1)
            categories, label = find_labels(categories, line[start+1:end])
            train_text.append(line[end+2:])
            train_label.append(label)
            count[label] += 1

    print('Labels:')
    index = np.argsort(count)
    for name, i in zip(np.array(categories)[index], count[index]):
        print('%4d' % i, name)

    for i, label in enumerate(train_label):
        binary_label = np.zeros(len(categories))
        for c in label:
            binary_label[c] = 1.
        train_label[i] = list(binary_label)
    
    return train_text, train_label, categories


def read_test(filename):
    
    test_text = []
    
    with open(filename, 'r', encoding='UTF-8') as f:
        
        f.readline()
        for line in f:
            start = line.find(',')
            test_text.append(line[start+1:])

    return test_text


def read_glove(filename):
    
    embedding_index = {}

    with open(filename, 'r', encoding='UTF-8') as f:
        
        i = 0
        for line in f:
            print('\r%d' % (i+1), end='', flush=True)
            values = line.split(' ')
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
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
    result = 2*((precision*recall)/(precision+recall))
    return result


EMBEDDING_DIM = 100

# argv: [1]train_data.csv [2]test_data.csv
def main():

    print('==================================================================')    
    print('Read train data.')
    train_text, train_label, categories = read_train(argv[1])
    print('Read test data.')
    test_text = read_test(argv[2])
    all_corpus = train_text + test_text
    print ('Find %d articles.' %(len(all_corpus)))

    print('==================================================================')    
    print('Tokenizer.')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_corpus)
    word_index = tokenizer.word_index
    print('Convert to index sequences.')    
    train_sequences = tokenizer.texts_to_sequences(train_text)
    test_sequences = tokenizer.texts_to_sequences(test_text)
    print('Pad sequences.')
    train_data = pad_sequences(train_sequences)
    train_label = np.array(train_label)
    MAX_SEQUENCE_LEN = train_data.shape[1]
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LEN)

    print('==================================================================')    
    print('Split validation.')
    (X_train, Y_train), (X_val, Y_val) = split_validation(train_data, train_label, 0.1)
    print('Shape of X:', train_data.shape)
    print('Shape of Y:', train_label.shape)
    print('Shape of X_train:', X_train.shape)
    print('Shape of Y_train:', Y_train.shape)
    print('Shape of X_val:', X_val.shape)
    print('Shape of Y_val:', Y_val.shape)
    print('Shape of test data:', test_data.shape)
    
    print('==================================================================')    
    print('Embedding layer.')
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    embedding_dict = read_glove('glove.6b.100d.txt')    
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    
    embedding_layer = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LEN, trainable=False)

    print('==================================================================')    
    print('Construct model.')
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(GRU(128, activation='tanh', dropout=0.1))
    model.add(Dense(256, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(38, activation='sigmoid'))
    model.summary()

    print('Compile model.')
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[f1_score])

    print('Train model.')
    EPOCHS = 1000
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
    check_point = ModelCheckpoint(filepath='model.h5', verbose=0, \
                                  save_best_only=True, save_weights_only=False, \
                                  monitor='val_loss', mode='min')
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),\
                        batch_size=128, epochs=EPOCHS, callbacks=[early_stopping, check_point])
    h = history.history
    idx = np.argmin(h['val_loss'])
    score = h['val_f1_score'][idx]
    print('selected epoch: %d, val_f1_score: %f' % (idx+1, score))

    print('==================================================================')    
    print('Save.')
    os.rename('model.h5', '{:.6f}'.format(score) + '.h5')
    np.savez('{:.6f}'.format(score) + '_history_%de.npz' % (idx+1), f1_score=h['f1_score'], val_h1_score=h['val_f1_score'])
    np.save('categories.npy', categories)
    np.save('{:.6f}'.format(score) + '_word_index.npy', word_index)


if __name__ == '__main__':
    main()
