import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
#categorical_crossentropy

def load_data():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	number = 10000   
	x_train = x_train[0:number]
	y_train = y_train[0:number]
        x_train = x_train.reshape(number, 28*28)
        x_test = x_test.reshape(x_test.shape[0], 28*28)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)
        x_train = x_train/255
        x_test = x_test/255
	#x_test=np.random.normal(x_test)
        return (x_train, y_train), (x_test, y_test)

(x_train,y_train),(x_test,y_test)=load_data()

model = Sequential()
model.add(Dense(input_dim=28*28,units=689,activation='relu'))
model.add(Dense(units=689,activation='relu'))
model.add(Dense(units=689,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=100,epochs=20)

score = model.evaluate(x_train,y_train)
print '\nTrain Acc:', score[1]
score = model.evaluate(x_test,y_test)
print '\nTest Acc:', score[1]

model2 = Sequential()
model2.add(Conv2D(25,(3,3),input_shape=(28,28,1)))
model2.add(MaxPooling2D((2,2)))
model2.add(Conv2D(50,(3,3)))
model2.add(MaxPooling2D((2,2)))
model2.add(Flatten())
model2.add(Dense(units=100,activation='relu'))
model2.add(Dense(units=10,activation='softmax'))
model2.summary()

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model2.fit(x_train,y_train,batch_size=100,epochs=20)

score = model2.evaluate(x_train,y_train)
print '\nTrain Acc:', score[1]
score = model2.evaluate(x_test,y_test)
print '\nTest Acc:', score[1]





