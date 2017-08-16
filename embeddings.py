import os

from keras import Sequential
from keras.layers import Embedding

from keras.preprocessing.sequence import skipgrams

def ptb(section='test.txt', directory='ptb/', padding='<EOS>'):
    with open(os.path.join(directory, section), 'rt') as fh:
        data = list(fh)
    data = [ ex.strip().split(' ') for ex in data ]
    data = [ ex + [padding] for ex in data ]
    vocab = set([ word for sent in data for word in sent ])
    return vocab, data

def text_to_sequence(texts, vocab, maxlen=30, pre=False, padding='<EOS>'):
    word_to_n = { word : i for i, word in enumerate(vocab, 1) }
    n_to_word = { i : word for word, i in word_to_n.items() }

    sequences = []
    for sent in texts:
        sequences.append([ word_to_n[word] for word in sent ])

    if pre:
        for sequence in sequences:
            sequence.insert(0, word_to_n[padding])

    #sequences = pad_sequences(sequences, maxlen)
    return sequences, word_to_n, n_to_word

def build(dim_embeddings):
    # inputs
    w_inputs = Input(shape=(1, ), dtype='int32')
    w = Embedding(V, dim_embedddings)(w_inputs)

    # context
    c_inputs = Input(shape=(1, ), dtype='int32')
    c  = Embedding(V, dim_embedddings)(c_inputs)
    o = Dot(axes=2)([w, c])
    o = Reshape((1,), input_shape=(1, 1))(o)
    o = Activation('sigmoid')(o)

    SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)

if __name__ == '__main__':
    print('Reading train data...')
    vocab_train, X_train = ptb(section='train.txt')
    vocab_valid, X_valid = ptb(section='valid.txt')
    vocab_test, X_test = ptb(section='test.txt')
    print('Done.')

    vocab = vocab_train.union(vocab_valid.union(vocab_test))
    X = X_train + X_valid + X_test

    print('Generating skipgrams...')
    sequences, word_to_n, n_to_word = text_to_sequence(X, vocab)
    X_skipgrams = skipgrams(sequences, len(vocab))
    print('Done.') 

    model = build(300)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fi
    
