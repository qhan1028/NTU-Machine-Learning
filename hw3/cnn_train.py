# ML 2017 hw3 Train CNN

import numpy as np
np.set_printoptions(precision = 6, suppress = True)
import csv
from sys import argv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint


SHAPE = 48
CATEGORY = 7

DATA_DIR = './data'
MODEL_DIR = './model'
HIS_DIR = './history'

READ_FROM_NPZ = 1
AUGMENT = 1

def read_train(filename):

    X, Y = [], []
    with open(filename, 'r', encoding='big5') as f:
        count = 0
        for line in list(csv.reader(f))[1:]:
            Y.append( float(line[0]) )
            X.append( [float(x) for x in line[1].split()] )
            count += 1
            print('\rX_train: ' + repr(count), end='', flush=True)
        print('', flush=True)

    return np.array(X), np_utils.to_categorical(Y, CATEGORY)

# argv: [1]train.csv
def main():

    print('============================================================')
    X, Y = [], []
    if READ_FROM_NPZ:
        print('Read from npz')
        data = np.load(DATA_DIR + '/data.npz')
        X = data['arr_0']
        Y = data['arr_1']
    else:
        print('Read train data')
        X, Y = read_train(argv[1])

    print('Reshape data')
    X = X/255
    X = X.reshape(X.shape[0], SHAPE, SHAPE, 1)

    print('============================================================')
    print('Construct model')
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

    print('Compile model')
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    print('============================================================')
    VAL = 2400
    BATCH = 128
    EPOCHS = 400
    if AUGMENT == 1: 
        print('Train with augmented data')
        datagen = ImageDataGenerator(vertical_flip=False, horizontal_flip=True, fill_mode='nearest', \
                                     height_shift_range=0.1, width_shift_range=0.1, rotation_range=20.)
        Xv = X[:VAL]
        Yv = Y[:VAL]
        datagen.fit(X[VAL:], seed=1019)
        cp = ModelCheckpoint('temp_model.h5', monitor='val_acc', verbose=0, save_best_only=True)
        history = model.fit_generator(datagen.flow(X[VAL:], Y[VAL:], batch_size=BATCH, seed=1019), callbacks=[cp],\
                                      samples_per_epoch=len(X[VAL:]), epochs=EPOCHS, verbose=1, validation_data=(Xv, Yv))

        print('============================================================')
        H = history.history
        best_val = '{:.6f}'.format(np.max(H['val_acc']))
        last_val = '{:.6f}'.format(H['val_acc'][-1])
        print('Best val: ' + best_val)
        print('Last val: ' + last_val)

        print('============================================================')
        print('Save best model')
        os.rename('temp_model.h5', MODEL_DIR + '/' + best_val + '.h5')
        print('Save last model')
        model.save(MODEL_DIR + '/' + last_val + '.h5')
        print('Save history')
        np.savez(HIS_DIR + '/' + best_val + '_history.npz', acc=H['acc'], val_acc=H['val_acc'])
    else:
        print('Train with raw data')
        es = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')
        model.fit(X, Y, batch_size=BATCH, epochs=EPOCHS, verbose=1, validation_split=0.1, callbacks=[es])

        print('============================================================')
        print('Evaluate train')
        score = model.evaluate(X, Y)
        score = '{:.6f}'.format(score[1])
        print('Train accuracy (all):', score)

        print('============================================================')
        print('Save model')
        model.save(MODEL_DIR + '/' + score + '.h5')


if __name__ == '__main__':
    main()
