from keras.models import Model
from keras.layers import Input
from keras.engine.topology import Layer
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

import keras.losses as losses
import keras.backend as K

def vae_lm(vocab_size=10000, input_length=30, embedding_dim=300, encoder_hidden_dim=100, \
        decoder_hidden_dim=100, latent_dim=50):

    inputs = Input(shape=(input_length,))       #n_0, n_1, ...
    tf = Input(shape=(input_length,))           #<EOS>, n_0, ...
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, \
            input_length=input_length, mask_zero=True)

    x = embedding_layer(inputs)
    x = GRU(encoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_state=True, name='encoder')(x)[-1]
    mu = Dense(latent_dim)(inputs)
    sigma = Dense(latent_dim, activation='softplus')(inputs)
    z = Lambda(lambda x: x[0] + x[1] * K.random_normal(shape=(latent_dim,), mean=0., stddev=1.))([mu, sigma])

    h_0 = Dense(decoder_hidden_dim)(z)
    x = embedding_layer(tf)
    x = GRU(decoder_hidden_dim, name='decoder', unroll=True, return_sequences=True)(x, initial_state=[h_0])
    x = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)

    #loss calculations
    dist_loss = Lambda(maximize_noise_loss, name='dist_loss')([mu, sigma])
    one_hot = Embedding(input_dim=vocab_size, output_dim=vocab_size, \
            embeddings_initializer='identity', mask_zero=True, trainable=False)(inputs)
    xent = Lambda(lambda x: K.sum(losses.categorical_crossentropy(x[0], x[1])), \
            output_shape=(1,))([one_hot, x])
    xent = CustomLossLayer(name='xent')([xent, dist_loss])

    encoder = Model(inputs=[inputs], outputs=[mu, sigma])
    model = Model(inputs=[inputs, tf], outputs=[xent, dist_loss])
    return encoder, model

def kl_loss(x):
    mu, sigma = x[0], x[1]
    return -0.5 * K.sum(1 + K.log(sigma) - K.square(mu) - sigma)

def maximize_noise_loss(x):
    mu, sigma = x[0], x[1]
    return K.sum(K.square(mu) + 1./K.square(sigma))

def identity(y_true, y_pred):
    return y_pred

def zero(y_true, y_pred):
    return K.zeros((1,))

class CustomLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomLossLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        xent = inputs[0]
        kl_loss = inputs[1]
        loss = xent + K.exp(-K.stop_gradient(xent)) * kl_loss
        self.add_loss(loss, inputs=inputs)
        return inputs[0]

if __name__ == '__main__':
    encoder, lm = vae_lm()
    lm.compile('rmsprop', loss=None)
