import os
import pickle
import numpy as np

from recurrentshop import RecurrentModel
from recurrentshop.cells import LSTMCell

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Lambda, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.merge import Concatenate, Add, Multiply
from keras.layers.wrappers import TimeDistributed

from keras.preprocessing.sequence import pad_sequences
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

def one_hot(seq):
    eye = np.eye(seq.max()+1)
    eye[0, 0] = 0
    ar = eye[seq]
    return ar

#Grammar as a Foreign Language
#Vinyals 2015 et al.
#TODO incorporate hidden state
def AttentionDecoder(hidden_dim, input_shape, initial_state=None):
    batch_size, input_length, input_dim = input_shape
    inputs = Input(shape=input_shape)
    a_tm1 = Input(shape=(input_length,))

    d_tm1 = Input(shape=(input_length,))
    h_tm1 = Input(shape=(hidden_dim,))
    c_tm1 = Input(shape=(hidden_dim,))

    x1 = Dense(hidden_dim, use_bias=False)(h_tm1)
    x2 = Dense(hidden_dim, use_bias=False)(d_tm1)
    x = Add()([x1, x2])
    x = Activation('tanh')(x)
    x = Dense(input_length)(x)

    a_t = Dense(input_length, activation='softmax')(x)
    d_t = Multiply()([a_t, x])
    _, h_t, c_t = LSTMCell(hidden_dim, input_dim=input_dim)([d_t, h_tm1, c_tm1])

    return RecurrentModel(input=inputs, input_shape=(input_length, input_dim), initial_states=[d_tm1, h_tm1, c_tm1], \
            output=h_t, final_states=[d_t, h_t, c_t], unroll=True, return_sequences=True)

def Seq2SeqAttention(input_length, vocab_size, encoder_hidden_dim=256, decoder_hidden_dim=256, \
        encoder_dropout=0.5, decoder_dropout=0.5, embedding_dim=128):
    inputs = Input(shape=(input_length,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, \
            input_length=input_length, mask_zero=True)(inputs)

    x1 = LSTM(encoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_sequences=True)(x)
    x2 = LSTM(encoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_sequences=True, go_backwards=True)(x)
    x = Concatenate()([x1, x2])
    encoding = Dropout(encoder_dropout)(x)          #(None, 50, 512)
    
    x = AttentionDecoder(decoder_hidden_dim, input_shape=K.int_shape(encoding), initial_state=None)(encoding)
    x = Dropout(decoder_dropout)(x)                 #(None, 50, 256)
    decoding = TimeDistributed(Dense(5, activation='softmax'))(x)
    return Model(inputs=inputs, outputs=decoding)

if __name__ == '__main__':
    print('Reading train/valid data...')
    in_vocab, X_train = ptb(section='wsj_24', directory='data/', column=0)
    out_vocab, y_train = ptb(section='wsj_24', directory='data/', column=1)

    in_vocab = read_vocab()
    in_vocab +=  [ '<unk>', '<EOS>' ]

    #TODO still need to read in validation and test data
    X_train_seq, word_to_n, n_to_word = text_to_sequence(X_train, in_vocab, maxlen=10)
    y_train_seq, _, _ = text_to_sequence(y_train, out_vocab, maxlen=10)
    y_train_seq = np.array([ one_hot(seq) for seq in y_train_seq ])
    print('Done.')

    print('Read in %d examples.' % len(X_train))
    print('Contains %d unique words.' % len(in_vocab))

    print('Building model...')
    optimizer = optimizers.Adam(lr=0.001)
    model = Seq2SeqAttention(input_length=10, vocab_size=len(in_vocab))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    print('Done.')

    print('Training model...')
    model.fit(X_train_seq, y_train_seq, batch_size=128, epochs=500)
    print('Done.')

    print('Saving models...')
    RUN = 'baseline'
    save_dir = os.path.join('runs/', RUN)
    model.save(os.path.join(save_dir, 'baseline.h5'))
    print('Done.')

    print('Testing...')
    print('Done.')
