import scipy.io
import scipy.misc
from PIL import Image
import sys
import matplotlib.image as mpimg

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

from encoder_decoder import *

class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""
    
    def __init__(self, texts, padding,
                 reverse=False, num_words=None):
        """
        :param texts: List of strings. This is the data-set.
        :param padding: Either 'post' or 'pre' padding.
        :param reverse: Boolean whether to reverse token-lists.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

        # Convert all texts to lists of integer-tokens.
        # Note that the sequences may have different lengths.
        self.tokens = self.texts_to_sequences(texts)

        if reverse:
            # Reverse the token-sequences.
            self.tokens = [list(reversed(x)) for x in self.tokens]
        
            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        # The number of integer-tokens in each sequence.
        self.num_tokens = [len(x) for x in self.tokens]

        # Max number of tokens to use in all sequences.
        # We will pad / truncate all sequences to this length.
        # This is a compromise so we save a lot of memory and
        # only have to truncate maybe 5% of all the sequences.
        self.max_tokens = np.mean(self.num_tokens) \
                          + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)
        # Pad / truncate all token-sequences to the given length.
        # This creates a 2-dim numpy matrix that is easier to use.
        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen=self.max_tokens,
                                           padding=padding,
                                           truncating=truncating)

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        
        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text
    
    def text_to_tokens(self, text, reverse=False, padding=False):
        """
        Convert a single text-string to tokens with optional
        reversal and padding.
        """

        # Convert to tokens. Note that we assume there is only
        # a single text-string so we wrap it in a list.
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            # Reverse the tokens.
            tokens = np.flip(tokens, axis=1)

            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        if padding:
            # Pad and truncate sequences to the given length.
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_tokens,
                                   padding='pre',
                                   truncating=truncating)

        return tokens





data = pd.read_csv('train_v2.csv')
y_train = data['tags']
mark_start = 'ssss '
mark_end = ' eeee'
def mark_captions(captions_list):
    captions_marked = [mark_start + caption + mark_end
                        for caption in captions_list]
                        
    
    return captions_marked

captions_train = mark_captions(y_train)
captions = TokenizerWrap(texts = captions_train,
                          padding = 'post',
                          reverse = False,
                          num_words = 10000)





def generate_caption(image_path, max_tokens=7):
    
    tst_img = []
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224,224))
    img_arr = np.asarray(image,dtype = 'float32')
    tst_img.append(img_arr)
    tst_img = np.array(tst_img)
    
    transfer_values = image_model.predict(tst_img)

    
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    
    token_int = 1

    
    output_text = ''

    count_tokens = 0

    while token_int != 2 and count_tokens < max_tokens:
        
        decoder_input_data[0, count_tokens] = token_int

        x_data = \
        {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data
        }

        
        decoder_output = decoder_model.predict(x_data)

        token_onehot = decoder_output[0, count_tokens, :]

        token_int = np.argmax(token_onehot)

        sampled_word = captions.token_to_word(token_int)

        output_text += " " + sampled_word

        count_tokens += 1

    output_tokens = decoder_input_data[0]

    plt.imshow(image)
    plt.show()
    
    #print("Predicted caption:")
    #print(output_text)
   # print()
    return output_text

#generate_caption("1.jpg")
