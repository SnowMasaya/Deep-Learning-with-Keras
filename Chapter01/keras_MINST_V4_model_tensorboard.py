from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from make_tensorboard_model import make_tensorboard
import tensorflow as tf


np.random.seed(1671)  # for reproducibility

# network and training
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10   # number of outputs = number of digits
OPTIMIZER = RMSprop()  # optimizer, explainedin this chapter
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2  # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3

# data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784
#
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# M_HIDDEN hidden layers
# 10 outputs
# final stage is softmax

with tf.name_scope('Model') as scope:
    model = Sequential()
    with tf.name_scope('Dense') as scope:
        model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))

    with tf.name_scope('Relu') as scope:
        model.add(Activation('relu'))

    with tf.name_scope('Dropuout') as scope:
        model.add(Dropout(DROPOUT))

    with tf.name_scope('Dense2') as scope:
        model.add(Dense(N_HIDDEN))

    with tf.name_scope('Relu2') as scope:
        model.add(Activation('relu'))

    with tf.name_scope('Dropuout2') as scope:
        model.add(Dropout(DROPOUT))

    with tf.name_scope('Dense3') as scope:
        model.add(Dense(NB_CLASSES))

    with tf.name_scope('Softmax') as scope:
        model.add(Activation('softmax'))

    model.summary()

callbacks = [make_tensorboard(set_dir_name='keras_MINST_V4')]

with tf.name_scope('ModelCompile') as scope:
    model.compile(loss='categorical_crossentropy',
                  optimizer=OPTIMIZER,
                  metrics=['accuracy'])

with tf.name_scope('TrainingModel') as scope:
    model.fit(X_train, Y_train,
              batch_size=BATCH_SIZE, epochs=NB_EPOCH,
              callbacks=callbacks,
              verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
