from keras.engine.topology import Layer

import keras.backend as K
import tensorflow as tf
K.set_session(tf.Session())

#custom loss
class CustomLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomLossLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        self.add_loss(inputs, inputs=inputs)
        return inputs

def neg_log_likelihood(y_true, y_pred):
    probs = tf.multiply(y_true, y_pred)
    probs = K.sum(probs, axis=-1)
    return K.sum(-K.log(K.epsilon() + probs))

#monitoring
def identity(y_true, y_pred):
    return y_pred

def zero(y_true, y_pred):
    return K.zeros((1,))
