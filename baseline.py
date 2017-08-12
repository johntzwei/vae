from utils import CustomLossLayer, neg_log_likelihood

from keras.models import Model
from keras.layers import Input
from keras.engine.topology import Layer
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

import keras.backend as K

def lm(vocab_size=10000, input_length=30, embedding_dim=300, encoder_hidden_dim=100, \
        decoder_hidden_dim=100, latent_dim=50):

    inputs = Input(shape=(input_length,))       #n_0, n_1, ...
    tf = Input(shape=(input_length,))           #<EOS>, n_0, ...
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, \
            input_length=input_length, mask_zero=True)

    x = embedding_layer(inputs)
    #problem is here!
    h_0 = GRU(encoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_state=True, name='encoder')(x)[-1]

    x = embedding_layer(tf)
    x = GRU(decoder_hidden_dim, name='decoder', unroll=True, return_sequences=True)(x, initial_state=[h_0])
    x = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)

    #loss calculations
    one_hot = Embedding(input_dim=vocab_size, output_dim=vocab_size, \
            embeddings_initializer='identity', mask_zero=True, trainable=False)(inputs)
    xent = Lambda(lambda x: neg_log_likelihood(x[0], x[1]), output_shape=(1,))([one_hot, x])
    x = CustomLossLayer()(xent)

    encoder = Model(inputs=[inputs], outputs=[h_0])
    model = Model(inputs=[inputs, tf], outputs=[x])
    return encoder, model

if __name__ == '__main__':
    encoder, lm = lm()
    lm.compile('rmsprop', loss=None)
