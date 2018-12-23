import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import os


from keras.models import Model,Sequential
from keras.layers import Input,Dense,GRU,LSTM,Embedding,Dropout,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D,ZeroPadding2D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard  
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences


from keras.optimizers import SGD
def VGG_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
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
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
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
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    
    if weights_path:
        model.load_weights(weights_path)

    return model

image_model = VGG_19(weights_path ='image_model.h5')

transfer_layer = image_model.get_layer('dense_2')
transfer_values_size = K.int_shape(transfer_layer.output)[1]
transfer_values_input = Input(shape=(transfer_values_size,),
                              name='transfer_values_input')




decoder_transfer_map = Dense(512,
                             activation='tanh',
                             name='decoder_transfer_map')
decoder_input = Input(shape=(None,),name = 'decoder_input')
decoder_embedding = Embedding(input_dim=10000,
                              output_dim = 128,
                              name = 'decoder_embedding')
decoder_gru1 =  GRU(512,
                    name = 'decoder_gru1',
                    return_sequences = True)
decoder_gru2 = GRU(512,
                   name = 'decoder_gru2',
                   return_sequences = True)
decoder_gru3 = GRU(512,
                   name = 'decoder_gru3',
                   return_sequences = True)
decoder_dense = Dense(1000,
                      activation = 'linear',
                      name = 'decoder_output')

def connect_decoder(transfer_values):
    initial_state = decoder_transfer_map(transfer_values)
    net = decoder_input
    net = decoder_embedding(net)
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)
    net = decoder_dense(net)
    return net

decoder_output = connect_decoder(transfer_values=transfer_values_input)
decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])

decoder_model.load_weights("decoder_model.h5")