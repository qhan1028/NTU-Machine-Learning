import os
import sys
import numpy as np
import random
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Input,Dense,Dropout,Flatten,Activation
from keras.layers import Convolution2D,MaxPooling2D
from keras.optimizers import SGD
from keras.optimizers import Adagrad, Adadelta
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator

VALIDATION=2000
BATCH_SIZE=256

def read_dataset(mode):
    
    datas = []
    
    if mode == 'train':
        with open(sys.argv[1]) as file:
            for line_id, line in enumerate(file):
                if line_id != 0:
                    label, feat = line.split(',')
                    feat = np.fromstring(feat, dtype=int, sep=' ')
                    feat = np.reshape(feat, (48, 48, 1))
                    feat = feat.astype('float32')/255.0 #normalization
                    datas.append((feat, int(label), line_id))
                    #left right flip
                    #feat_lr = np.fliplr(feat)
                    #datas.append((feat_lr, int(label), line_id))
                    #up down flip
                    #feat_ud = np.flipud(feat)
                    #datas.append((feat_ud, int(label), line_id))
        
        feats,labels, line_ids = zip(*datas) 
        feats = np.asarray(feats)
        labels = to_categorical(np.asarray(labels,dtype=np.int32), 7)
        return feats, labels, line_ids

    elif mode == 'test':
        with open(sys.argv[2]) as file:
            for line_id, line in enumerate(file):
                if line_id != 0:
                    _, feat = line.split(',')
                    feat = np.fromstring(feat, dtype=int, sep=' ')
                    feat = np.reshape(feat, (48, 48, 1))
                    feat = feat.astype('float32')/255.0 #normalization
                    datas.append(feat)
        return np.asarray(datas),0,0

def build_model():
    model = Sequential()
    
    # CNN part (you can repeat this part several times)
    model.add(Convolution2D(32,(3,3),input_shape=(48,48,1), activation='relu'))
    model.add(Convolution2D(32,(3,3), activation="relu"))
    #model.add(Convolution2D(32,(3,3), activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64,(3,3), activation="relu", padding='same'))
    model.add(Convolution2D(64,(3,3), activation="relu", padding='same'))
    #model.add(Convolution2D(64,(3,3), activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128,(3,3), activation="relu"))
    model.add(Convolution2D(128,(3,3), activation="relu"))
    #model.add(Convolution2D(128,(3,3), activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully connected part
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(7)) # ta use a variable nb_class
    model.add(Activation('softmax'))
    #opt = SGD(lr=0.01,decay=0.0)
    #opt = Adagrad(lr=0.0001,epsilon=1e-06)
    opt = Adadelta()
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary() # show the whole model in terminal
    return model


def main():
    emotion_classifier = build_model()
    train_feats, train_labels, _ = read_dataset('train')


    datagen = ImageDataGenerator(
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = True,
        fill_mode='nearest')
   
    datagen.fit(train_feats[:-VALIDATION,:,:,:], seed=512)
    history = emotion_classifier.fit_generator(datagen.flow(x = train_feats[:-VALIDATION], y = train_labels[:-VALIDATION], batch_size=BATCH_SIZE), steps_per_epoch=(len(train_feats) - VALIDATION) / 100, epochs=30, validation_data=(train_feats[-VALIDATION:], train_labels[-VALIDATION:]))
    
    emotion_classifier.summary()
    emotion_classifier.save('model_img.h5')
    h = history.history
    np.savez("history_img.npz", h["acc"], h["val_acc"], h["loss"], h["val_loss"])
    test_feats,_,_ = read_dataset('test')
    ans = emotion_classifier.predict_classes(test_feats,batch_size=64)

    with open('Answer_img','w') as f:
        f.write('id,label\n')
        for idx,a in enumerate(ans):
            f.write('{},{}\n'.format(idx,a))


if __name__ == "__main__":
    main()

