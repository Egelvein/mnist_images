import keras
from keras import layers
from keras import models
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = models.Sequential()
model.add(keras.Input(shape=input_shape))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop', 
              metrics=['acc'])
history = model.fit(x_train,
                    y_train, 
                    epochs = 10, 
                    batch_size = 1024)

data_test = model.predict(x_test)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_acc:', test_acc)
print('train_loss:', test_loss)

num = 50 #Choose any number in range (10000)
number = y_test[num]
plt.plot(data_test[num])
for i in range(10):
    if number[i] == 1:
        print(i)
print(y_test[num])
