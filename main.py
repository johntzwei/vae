import os

from keras.models import Model
from keras.layers import Input, Layer
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences

import keras.losses as losses
import keras.backend as K

import numpy as np
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)

def ptb(section='test.txt', directory='ptb/', padding='<EOS>'):
    with open(os.path.join(directory, section), 'rt') as fh:
        data = list(fh)
    data = [ ex.strip().split(' ') for ex in data ]
    data = [ ex + [padding] for ex in data ]
    vocab = set([ word for sent in data for word in sent ])
    return vocab, data

def text_to_sequence(texts, vocab, maxlen=30, pre=False, padding='<EOS>'):
    word_to_n = { word : i for i, word in enumerate(vocab) }
    n_to_word = { i : word for word, i in word_to_n.items() }

    sequences = []
    for sent in texts:
        sequences.append([ word_to_n[word] for word in sent ])

    if pre:
        for sequence in sequences:
            sequence.insert(0, word_to_n[padding])

    sequences = pad_sequences(sequences, maxlen)
    return sequences, word_to_n, n_to_word

def vae_lm(input_length=30, embedding_dim=300, encoder_hidden_dim=100, \
        decoder_hidden_dim=100, latent_dim=50):

    inputs = Input(shape=(input_length,), name='main')      #n_0, n_1, ...
    tf = Input(shape=(input_length,), name='tf')            #<EOS>, n_0, ...
    embedding_layer = Embedding(input_dim=len(vocab), output_dim=embedding_dim, \
            input_length=input_length, mask_zero=True)

    x = embedding_layer(inputs)
    x = LSTM(encoder_hidden_dim, input_shape=(input_length, embedding_dim), name='encoder')(x)

    mu = Dense(latent_dim)(x)
    log_var = Dense(latent_dim, activation='softplus')(x)
    epsilon = K.random_normal(shape=(latent_dim,), mean=0., stddev=1.)
    z = Lambda(lambda x: x[0] + K.exp(x[1] / 2) * epsilon, output_shape=(latent_dim,))([mu, log_var])

    h_0 = Dense(decoder_hidden_dim)(z)
    x = embedding_layer(tf)
    #need to figure out how to initialize the cell state as trainable weights
    x = LSTM(decoder_hidden_dim, name='decoder', return_sequences=True)(x, initial_state=[h_0, h_0])
    x = TimeDistributed(Dense(len(vocab), activation='softmax'))(x)

    #loss calculations
    kl_loss = Lambda(lambda x: -0.5 * K.sum(1 + x[1] - K.square(x[0]) - K.exp(x[1])), \
            output_shape=(1,), name='kl_loss')([mu, log_var])
    one_hot = Embedding(input_dim=len(vocab), output_dim=len(vocab), \
            embeddings_initializer='identity', mask_zero=True, trainable=False)(inputs)
    xent = Lambda(lambda x: K.sum(losses.categorical_crossentropy(x[0], x[1])), \
            output_shape=(1,), name='xent')([one_hot, x])
    outputs = CustomLossLayer(name='y')([x, xent, kl_loss])

    model = Model(inputs=[inputs, tf], outputs=[outputs, xent, kl_loss])
    return model

class CustomLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        self.supports_masking = True
        super(CustomLossLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        xent = inputs[1]
        kl_loss = inputs[2]

        loss = xent #+ kl_loss
        self.add_loss(loss, inputs=inputs)
        return inputs[0]

    def compute_mask(self, inputs, mask):
        return mask

def zero(y_true, y_pred):
    return K.zeros((1,))

def KL_Divergence(y_true, y_pred):
    return y_pred

if __name__ == '__main__':
    print('Reading train data...')
    vocab_train, X_train = ptb(section='train.txt')
    vocab_valid, X_valid = ptb(section='valid.txt')
    vocab_test, X_test = ptb(section='test.txt')

    vocab = vocab_train.union(vocab_valid.union(vocab_test))
    X = X_train + X_valid + X_test

    print('Read in %d examples.' % len(X))
    print('Contains %d unique words.' % len(vocab))

    sequences, word_to_n, n_to_word = text_to_sequence(X, vocab)
    tf_sequences, _, _ = text_to_sequence(X, vocab, pre=True)

    print('Building model...')
    model = vae_lm()
    model.compile(optimizer='rmsprop', loss={'kl_loss':zero}, \
            metrics={'kl_loss':KL_Divergence})

    print('Training model...')
    model.fit([sequences, tf_sequences], sequences, batch_size=32, epochs=100)
