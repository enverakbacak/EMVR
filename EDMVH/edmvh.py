#coding=utf-8
import cv2
import numpy as np
import sys,os
import time
import matplotlib
import scipy.io
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras import optimizers
import keras
from keras.models import Input,Model,InputLayer
from keras.layers import Dense, Dropout, Flatten, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import MobileNetV2
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize   # for resizing images
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
import keras.backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model

Y = pd.read_csv(r'Y3.csv') # Video files labels, one hot encoded
Y.shape
print(Y.shape[:])


X = np.load(open('NP3.npy')) # Join video frames in to one NPY file , see deneme.py
X.shape
print(X.shape[:])   # (number of videos, number frames in each video)
#X = (X - np.mean(X,axis = 1)) / np.std(X, axis = 1) #Normalize the data
X = X.reshape(X.shape[0], X.shape[1] , X.shape[2] * X.shape[3] * X.shape[4]) # (number of videos, number frames in each video, 7x7x512)
#X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2 ,random_state=43)



batch_size = 16
epochs = 20
hash_bits = 128

visible   = Input(shape = (X.shape[1] ,X.shape[2]))
blstm_1   = Bidirectional(LSTM(512, dropout=0.1, recurrent_dropout=0.5, input_shape=(X.shape[1], X.shape[2]), return_sequences = True  ))(visible)
blstm_2   = Bidirectional(LSTM(512, dropout=0.1, recurrent_dropout=0.5, input_shape=(X.shape[1], X.shape[2]), return_sequences = False ))(blstm_1)
batchNorm = BatchNormalization()(blstm_2)
Dense_1   = Dense(256)(batchNorm)
Dense_2   = Dense(hash_bits, activation = 'sigmoid' )(Dense_1)
#batchNorm = BatchNormalization()(Dense_2)
Dense_3   = Dense(7, activation='sigmoid')(Dense_2)
model     = Model(input = visible, output=Dense_3)
print(model.summary())
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



import keras.backend as K
# e = 0.5
def c_loss(noise_1, noise_2):
    def loss(y_true, y_pred):
         return (K.binary_crossentropy(y_true, y_pred))
         #return (K.binary_crossentropy(y_true, y_pred) + (1/hash_bits) * (K.sum((noise_1 - noise_2)**2) )  )  
         return ( (K.binary_crossentropy(y_true, y_pred)) + (K.sum(K.binary_crossentropy(noise_1, noise_2) )) * (1/hash_bits)   )
    return loss

from keras.optimizers import SGD
sgd = SGD(lr=0.01, decay = 1e-6, momentum=0.9, nesterov=True)



model.compile(loss = c_loss(noise_1 = tf.to_float(Dense_2 > 0.5 ), noise_2 = Dense_2 ),  optimizer=sgd, metrics=['accuracy']) 
#history = model.fit(X_train, Y_train, shuffle=True, batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(X_valid, Y_valid) )
history = model.fit(X, Y, shuffle=True, batch_size=batch_size,epochs=epochs,verbose=1 )


model_json = model.to_json()
with open("models/128_3_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/128_3_weights.h5")


params = {'legend.fontsize': 36,
          'legend.handlelength': 4,}
plt.rcParams.update(params)

matplotlib.rcParams.update({'font.size': 36})
plt.plot(history.history['acc'] , linewidth=5, color="green")
plt.plot(history.history['val_acc'], linestyle='--',  linewidth=5, color="red")
#plt.title('model accuracy' , fontsize=32)
plt.ylabel('Accuracy' , fontsize=40)
plt.xlabel('The number of epochs' , fontsize=36)
plt.legend( ['train', 'validation'], loc='best')
plt.show()
# summarize history for loss
matplotlib.rcParams.update({'font.size': 36})
plt.plot(history.history['loss'], linewidth=5, color="green")
plt.plot(history.history['val_loss'], linestyle='--', linewidth=5, color="red")
#plt.title('model loss' , fontsize=32)
plt.ylabel('Loss' , fontsize=40)
plt.xlabel('The number of epochs' , fontsize=36)
plt.legend( ['train', 'validation'], loc='best')
plt.show()


score = model.evaluate(X_train, Y_train)
print(model.metrics_names)
print(score)

score = model.evaluate(X_valid, Y_valid)
print(model.metrics_names)
print(score)

score = model.evaluate(X, Y)
print(model.metrics_names)
print(score)
