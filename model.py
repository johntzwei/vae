from keras.models import Model
from keras.layers import Input, Layer
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU

import keras.losses as losses
import keras.backend as K

def vae_lm(vocab_size=10000, input_length=30, embedding_dim=300, encoder_hidden_dim=100, \
        decoder_hidden_dim=100, latent_dim=50):

    inputs = Input(shape=(input_length,), name='main')      #n_0, n_1, ...
    tf = Input(shape=(input_length,), name='tf')            #<EOS>, n_0, ...
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, \
            input_length=input_length, mask_zero=True)

    x = embedding_layer(inputs)
    x = GRU(encoder_hidden_dim, input_shape=(input_length, embedding_dim), name='encoder')(x)

    mu = Dense(latent_dim, name='mu')(x)
    log_var = Dense(latent_dim, name='log_var')(x)
    epsilon = Lambda(lambda x: K.random_normal(shape=(latent_dim,), mean=0., stddev=1.))(x)
    z = Lambda(lambda x: x[0] + K.exp(x[1] / 2) * x[2], output_shape=(latent_dim,))([mu, log_var, epsilon])

    h_0 = Dense(decoder_hidden_dim)(z)
    x = embedding_layer(tf)
    #need to figure out how to initialize the cell state as trainable weights
    x = GRU(decoder_hidden_dim, name='decoder', return_sequences=True)(x, initial_state=[h_0])
    x = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)

    #loss calculations
    kl_loss = Lambda(lambda x: -0.5 * K.sum(1 + x[1] - K.square(x[0]) - K.exp(x[1])), \
            output_shape=(1,), name='kl_loss')([mu, log_var])
    one_hot = Embedding(input_dim=vocab_size, output_dim=vocab_size, \
            embeddings_initializer='identity', mask_zero=True, trainable=False)(inputs)
    xent = Lambda(lambda x: K.sum(losses.categorical_crossentropy(x[0], x[1])), \
            output_shape=(1,), name='xent')([one_hot, x])
    outputs = CustomLossLayer(name='y')([x, xent, kl_loss])

    model = Model(inputs=[inputs, tf], outputs=[outputs, xent, kl_loss])
    encoder = Model(inputs=[inputs], outputs=[mu, log_var])
    return encoder, model

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
