from keras.engine.topology import Layer
from keras.layers.merge import multiply
import keras.backend as K

#custom loss
class CustomLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomLossLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        self.add_loss(inputs, inputs=inputs)
        return inputs

def neg_log_likelihood(y_true, y_pred):
    probs = multiply([y_true, y_pred])
    probs = K.sum(probs, axis=-1)
    return 1e-06 + K.sum(-K.log(probs))

#monitoring
def identity(y_true, y_pred):
    return y_pred

def zero(y_true, y_pred):
    return K.zeros((1,))
