from keras.models import Sequential
from keras.layers.core import Dense,Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
import json
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

np.set_printoptions(precision=8,suppress=True,linewidth=1000,threshold=np.nan)
train_raw = open('train.csv','r')
train = train_raw.readlines()
train.pop(0)
for i in range(len(train)):
	train[i]=train[i].replace(',',' ')
	train[i]=train[i].replace('\n','')


train_data=np.empty((len(train),48,48),dtype="float32")
label=np.empty((len(train),),dtype="uint8")
for i in range(len(train)):
	tmplist=list(map(int,train[i].split(" ")))
	label[i]=tmplist[0]
	train_data[i]=np.reshape(tmplist[1:],(48,48))

train_data=np.reshape(train_data,(len(train),48,48,1))
print(np.shape(train_data))



label = np_utils.to_categorical(label,7)
validation_X=train_data[-3000:]
train_data=train_data[:-3000]
validation_Y=label[-3000:]
label=label[:-3000]

model=Sequential()

'''model.add(Convolution2D(32,3,3,input_shape=(48,48,1)))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.25))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
#model.add(Dropout(0.25))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.25))

model.add(Convolution2D(128,3,3))
model.add(Activation('relu'))
#model.add(Dropout(0.25))

model.add(Convolution2D(128,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(output_dim=100))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Dense(output_dim=100))
model.add(Activation('relu'))
model.add(Dense(output_dim=7))
model.add(Activation('softmax'))'''

model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()
datagen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,horizontal_flip=True,fill_mode='nearest',rotation_range=0.1)
datagen.fit(train_data,seed=256)
model.fit_generator(datagen.flow(train_data,label,batch_size=256),steps_per_epoch=len(train_data)/100,verbose=1,epochs=20,validation_data=(validation_X,validation_Y))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

test_raw = open('test.csv','r')
test = test_raw.readlines()
test.pop(0)
for i in range(len(test)):
	test[i]=test[i].replace(',',' ')
	test[i]=test[i].replace('\n','')


test_data=np.empty((len(test),48,48),dtype="float32")
label=np.empty((len(test),),dtype="uint8")
for i in range(len(test)):
	tmplist=list(map(int,test[i].split(" ")))
	label[i]=tmplist[0]
	test_data[i]=np.reshape(tmplist[1:],(48,48))

test_data=np.reshape(test_data,(len(test),48,48,1))

answer=model.predict(test_data)
answer=np.argmax(answer,axis=1)

result = open("result", 'w')
result.write('id,label\n')
for i in range(len(answer)):
	result.write('{},{}\n'.format(i, answer[i]))
result.close()


















