import os

import model

from keras.preprocessing.sequence import pad_sequences

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
    vae_lm = model.vae_lm(vocab_size=len(vocab))
    vae_lm.compile(optimizer='rmsprop', loss={'kl_loss':model.zero}, \
            metrics={'kl_loss':model.KL_Divergence})

    print('Training model...')
    vae_lm.fit([sequences, tf_sequences], sequences, batch_size=32, epochs=1)
    vae.save('models/vae_lm.h5')
