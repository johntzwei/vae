import os
import pickle
import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Dropout, Lambda, \
        RepeatVector, Permute
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.merge import Concatenate, Add, Multiply
from keras.layers.wrappers import TimeDistributed

from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras import optimizers
import keras.backend as K

def ptb(section='test.txt', directory='ptb/', padding='<EOS>', column=0):
    with open(os.path.join(directory, section), 'rt') as fh:
        data = [ i.split('\t')[column] for i in fh ]
    data = [ ex.strip().split(' ') for ex in data ]
    data = [ ex + [padding] for ex in data ]
    vocab = set([ word for sent in data for word in sent ])
    return vocab, data

def read_vocab(vocab='vocab', directory='data/'):
    with open(os.path.join(directory, vocab), 'rt') as fh:
        vocab = [ i.strip().split('\t')[0] for i in fh ]
    return vocab

def text_to_sequence(texts, vocab, maxlen=30, padding='<EOS>'):
    word_to_n = { word : i for i, word in enumerate(vocab, 1) }
    n_to_word = { i : word for word, i in word_to_n.items() }

    sequences = []
    for sent in texts:
        sequences.append([ word_to_n[word] for word in sent ])

    sequences = pad_sequences(sequences, maxlen)
    return sequences, word_to_n, n_to_word

def one_hot(seqs):
    n_values = np.max(seqs) + 1
    eye = np.eye(n_values)
    eye[0, 0] = 0
    return np.array([ eye[seq] for seq in seqs ])

class AttentionLSTM(LSTM):
    def __init__(self, output_dim, output_length=100, **kwargs):
        super(AttentionLSTM, self).__init__(output_dim, **kwargs)
        self.output_length = output_length

    def build(self, input_shape):
        super(AttentionLSTM, self).build(input_shape)

        self.states = [None, None, None]
        self.W_1 = Dense(self.units, use_bias=False)
        self.W_2 = Dense(self.units, use_bias=False)
        self.V = self.add_weight(shape=(self.units,), \
                initializer='glorot_uniform', name='v')

    def get_constants(self, inputs, training=None):
        constants = super(AttentionLSTM, self).get_constants(inputs, training=training)
        constants.insert(0, inputs)
        return constants

    def step(self, inputs2, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        d_tm1 = states[2]
        inputs = states[3]
        batch_size = K.int_shape(inputs)[0]
        samples = K.int_shape(inputs)[1]

        x1 = TimeDistributed(self.W_1)(inputs)
        x2 = self.W_2(d_tm1)
        x = Add()([x1, x2])         #broadcast
        x = Activation('tanh')(x)

        #dot product of v with each row
        x = self.V * x              #broadcast
        x = K.sum(x, axis=-1)

        x = Activation('softmax')(x)
        a_t = K.expand_dims(x, axis=-1)
        x = inputs * a_t          #broadcast
        d_t = K.sum(x, axis=-2)

        #d_t = self.preprocess_input(d_t)
        lstm_states = [h_tm1, c_tm1] + list(states[4:])
        h, (h, c) = super(AttentionLSTM, self).step(d_t, lstm_states)      #pass in only lstm states
        return h, [h, c, d_t]

    def compute_output_shape(self, input_shape):
        input_shape = (input_shape[0], self.output_length, input_shape[2])
        return super(AttentionLSTM, self).compute_output_shape(input_shape)

#Grammar as a Foreign Language
#Vinyals 2015 et al.
#TODO incorporate hidden state
def Seq2SeqAttention(input_length, output_length, vocab_size, out_vocab_size, 
        encoder_hidden_dim=256, decoder_hidden_dim=256, encoder_dropout=0.5, decoder_dropout=0.5, embedding_dim=128):
    inputs = Input(shape=(input_length,))
    x = Embedding(input_dim=vocab_size+1, output_dim=embedding_dim, \
            input_length=input_length, mask_zero=True)(inputs)

    x1 = LSTM(encoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_sequences=True)(x)
    x2 = LSTM(encoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_sequences=True, go_backwards=True)(x)
    x = Add()([x1, x2])
    encoding = Dropout(encoder_dropout)(x)          #(None, 50, 512)
    
    x = AttentionLSTM(decoder_hidden_dim, output_length=output_length, return_sequences=True, \
            implementation=1)(encoding)
    x = Dropout(decoder_dropout)(x)                 #(None, 50, 256)
    outputs = TimeDistributed(Dense(5, activation='softmax'))(x)
    return Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    print('Reading vocab...')
    in_vocab = read_vocab()
    in_vocab +=  [ '<unk>', '<EOS>' ]

    out_vocab = ['<EOS>', '(', ')', '<TOK>' ]
    print('Done.')

    print('Reading train/valid data...')
    _, X_train = ptb(section='wsj_2-21', directory='data/', column=0)
    _, y_train = ptb(section='wsj_2-21', directory='data/', column=1)
    X_train_seq, word_to_n, n_to_word = text_to_sequence(X_train, in_vocab, maxlen=20)
    y_train_seq, _, _ = text_to_sequence(y_train, out_vocab, maxlen=20)

    _, X_valid = ptb(section='wsj_24', directory='data/', column=0)
    _, y_valid = ptb(section='wsj_24', directory='data/', column=1)
    X_valid_seq, word_to_n, _ = text_to_sequence(X_valid, in_vocab, maxlen=20)
    y_valid_seq, _, _ = text_to_sequence(y_valid, out_vocab, maxlen=20)
    print('Done.')

    print('Contains %d unique words.' % len(in_vocab))
    print('Read in %d examples.' % len(X_train))

    print('Building model...')
    optimizer = optimizers.RMSprop(lr=0.001)
    model = Seq2SeqAttention(input_length=20, output_length=20, vocab_size=len(in_vocab), out_vocab_size=len(out_vocab))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    plot_model(model, to_file='model.png')
    print('Done.')

    print('Training model...')
    model.fit(X_train_seq, one_hot(y_train_seq), validation_data=(X_valid_seq, one_hot(y_valid_seq)), batch_size=64, epochs=1)
    print('Done.')

    print('Saving models...')
    RUN = 'baseline'
    save_dir = os.path.join('runs/', RUN)
    model.save(os.path.join(save_dir, 'baseline.h5'))
    print('Done.')

    print('Testing...')
    print('Done.')
