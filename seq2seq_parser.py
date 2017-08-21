import os
import pickle

import model
from utils import zero, identity

from keras.preprocessing.sequence import pad_sequences
from keras import optimizers

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

    sequences = pad_sequences(sequences, maxlen)
    return sequences, word_to_n, n_to_word

if __name__ == '__main__':
    print('Reading train data...')
    vocab_train, X_train = ptb(section='train.txt')
    vocab_valid, X_valid = ptb(section='valid.txt')
    vocab_test, X_test = ptb(section='test.txt')

    vocab = vocab_train.union(vocab_valid.union(vocab_test))
    X = X_train + X_valid + X_test

    vocab = vocab_valid
    X = X_valid[:1000]

    sequences, word_to_n, n_to_word = text_to_sequence(X, vocab)
    tf_sequences, _, _ = text_to_sequence(X, vocab, pre=True)

    print('Read in %d examples.' % len(X))
    print('Contains %d unique words.' % len(vocab))

    print('Building model...')
    encoder, vae_lm = model.vae_lm(vocab_size=len(vocab)+1, latent_dim=2)
    trainer = optimizers.RMSprop(lr=0.001)
    vae_lm.compile(optimizer=trainer, loss={'xent':zero, 'dist_loss':zero}, \
            metrics={'xent':identity, 'dist_loss':identity})
    print('Done.')

    print('Training model...')
    vae_lm.fit([sequences, tf_sequences], [sequences, tf_sequences], batch_size=32, epochs=500)
    print('Done.')

    RUN = 'preliminary'
    save_dir = os.path.join('runs/', RUN)

    print('Saving models...')
    vae_lm.save(os.path.join(save_dir, 'vae_lm.h5'))
    encoder.save(os.path.join(save_dir, 'encoder.h5'))
    print('Done.')

    print('Saving vocabularies...')
    pickle.dump(word_to_n, open(os.path.join(save_dir, 'word_to_n.pkl'),'wb'))
    pickle.dump(n_to_word, open(os.path.join(save_dir, 'n_to_word.pkl'),'wb'))
    print('Done.')

    print('Generating and saving embeddings...')
    embeddings = encoder.predict(sequences)
    pickle.dump(list(zip(X, embeddings[0])), open(os.path.join(save_dir, 'embeddings.pkl'),'wb'))
    print('Done.')

