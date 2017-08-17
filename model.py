from utils import CustomLossLayer, neg_log_likelihood

from keras.models import Model
from keras.layers import Input
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout, Lambda, RepeatVector, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.merge import Concatenate

import keras.initializers
import keras.backend as K
import tensorflow as tf

def vae_lm(vocab_size=10000, input_length=30, embedding_dim=300, encoder_hidden_dim=100, \
        decoder_hidden_dim=100, latent_dim=50, encoder_dropout=0.5, decoder_dropout=0.5):

    inputs = Input(shape=(input_length,))       #n_0, n_1, ...
    tf = Input(shape=(input_length,))           #<EOS>, n_0, ...

    #embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, \
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=vocab_size, \
            embeddings_initializer='identity', trainable=False, \
            input_length=input_length, mask_zero=True)

    x = embedding_layer(inputs)
    x = LSTM(encoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_state=True, name='encoder')(x)[-1]
    x = Dropout(encoder_dropout)(x)

    mu = Dense(latent_dim)(inputs)
    sigma = Dense(latent_dim, activation='softplus')(inputs)

    z = Lambda(lambda x: x[0] + x[1] * K.random_normal(shape=(latent_dim,), mean=0., stddev=1.))([mu, sigma])
    z = RepeatVector(input_length)(z)

    #sum of sentence word embeddings
    x = embedding_layer(tf)
    x = Lambda(lambda x: K.sum(x, axis=-1), output_shape=(input_length,))(x)
    x = RepeatVector(input_length)(x)
    x = Concatenate()([x, z])
    idx = Lambda(lambda x: K.cast(K.arange(start=0, stop=input_length), 'float32'), output_shape=(input_length,))(x)
    idx = Reshape((input_length, 1))(idx)
    #idx = Lambda(lambda x: K.eye(input_length), output_shape=(input_length, input_length))(x)
    #idx = Reshape((input_length, input_length))(idx)
    x = Concatenate(axis=2)([x, idx])
    x = TimeDistributed(Dense(decoder_hidden_dim, activation='tanh'))(x)
    x = TimeDistributed(Dropout(decoder_dropout))(x)
    x = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)

    #loss calculations
    dist_loss = Lambda(box_loss, name='dist_loss')([mu, sigma])
    one_hot = Embedding(input_dim=vocab_size, output_dim=vocab_size, \
            embeddings_initializer='identity', mask_zero=True, trainable=False)(inputs)
    xent = Lambda(lambda x: neg_log_likelihood(x[0], x[1]), output_shape=(1,), name='xent')([one_hot, x])
    loss = Lambda(exp_annealing, output_shape=(1,))([xent, dist_loss])
    x = CustomLossLayer()(loss)

    encoder = Model(inputs=[inputs], outputs=[mu, sigma])
    model = Model(inputs=[inputs, tf], outputs=[xent, dist_loss, x])
    return encoder, model

#distribution losses
def kl_loss(x):
    mu, sigma = x[0], x[1]
    return -0.5 * K.sum(1 + K.log(K.epsilon()+sigma) - K.square(mu) - sigma)

def maximize_noise_loss(x):
    mu, sigma = x[0], x[1]
    return K.sum(K.square(mu) - K.square(sigma))

def box_loss(x, box=100.):
    mu, sigma = x[0], x[1]
    return K.sum(-K.square(sigma)) + K.square(K.minimum(0., mu-box)) + K.square(K.maximum(0., box+mu))

#annealing
def exp_annealing(x):
    return x[0] + K.exp(-K.stop_gradient(x[0])) * x[1]

def add_losses(x):
    return x[0] + x[1]

if __name__ == '__main__':
    encoder, lm = vae_lm()
    lm.compile('rmsprop', loss=None)
