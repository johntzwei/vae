import os
import pickle

import model

from nltk import CFG
from nltk.parse.generate import generate

from keras.preprocessing.sequence import pad_sequences

#binary branching parens language
grammar5 = CFG.fromstring( \
        '''
        S -> '(' A A ')'
        A -> '*' | S
        '''
    )
 
def base_cases(grammar, depth):
    cases = []
    for s in generate(grammar, depth=depth,):
        s = ''.join(s)
        cases.append(s)
    return cases

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
    X = base_cases(grammar5, 9)
    vocab = list('()*') + ['<EOS>']

    sequences, word_to_n, n_to_word = text_to_sequence(X, vocab)
    tf_sequences, _, _ = text_to_sequence(X, vocab, pre=True)

    print('Read in %d examples.' % len(X))
    print('Contains %d unique words.' % len(vocab))

    print('Building model...')
    encoder, vae_lm = model.vae_lm(vocab_size=len(vocab))
    vae_lm.compile(optimizer='rmsprop', loss={'kl_loss':model.zero}, \
            metrics={'kl_loss':model.KL_Divergence})
    print('Done.')

    print('Training model...')
    vae_lm.fit([sequences, tf_sequences], sequences, batch_size=32, epochs=100)
    print('Done.')

    RUN = 'cfgs'
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

