import re
from nltk.corpus import BracketParseCorpusReader

wsj = '/data/penn_tb_3.0/TREEBANK_3/PARSED/MRG/WSJ/'
file_pattern = r".*/WSJ_.*\.MRG"
ptb = BracketParseCorpusReader(wsj, file_pattern)

print('Gathered %d files...' % len(ptb.fileids()))

TRAIN_FILE = 'data/wsj_2-21'
TEST_FILE = 'data/wsj_23'
DEV_FILE = 'data/wsj_24'
SECTIONS = [ (2, 21), (23, 23), (24, 24) ]

#this normalization gets applied to every word
def normalize(word):
    try:
        if float(word):
            return 'N'
    except:
        pass

    word = word.lower()
    return word

def linearize(tree, label=False):
    if label:
        pass
    else:
        tree.set_label('')
        for subtree in tree.subtrees():
            subtree.set_label('')

    for subtree in tree.subtrees(filter=lambda x: x.height() == 2):
        leaf = normalize(subtree[0])
        if leaf not in vocab:
            subtree[0] = '<unk>'
        else:
            subtree[0] = leaf

    lin = tree.pformat(margin=10000, nodesep='', parens=['(', ' )'])
    lin = re.sub(r'\s+', ' ', lin)
    return lin

#get vocab
VOCAB_FILE = 'data/vocab'
print('Generating vocabulary...')
vocab = {}
for sections in SECTIONS:
    for section in range(sections[0], sections[1]+1):
        fileids = [ i for i in ptb.fileids() if i.startswith(str(section).zfill(2)) ]

        for sent in ptb.sents(fileids):
            for word in sent:
                word = normalize(word)
                counter = vocab.get(word, 0)
                vocab[word] = counter + 1

#filter out vocab
vocab = list(vocab.items())
vocab.sort(key=lambda x: -x[1])
vocab = vocab[:10000-4]
print('Vocabulary has %d words.' % len(vocab))
print('Top 10 words in vocabulary: %s' % str(vocab[:10]))

print('Writing vocab to file...')
h = open(VOCAB_FILE, 'wt')
for word, freq in vocab:
    h.write('%s\t%s\n' % (word, freq))
h.close()
print('Done.')

vocab = set([ i[0] for i in vocab ])
print('Done.')

#write trees to files
for fn, sections in zip([ TRAIN_FILE, TEST_FILE, DEV_FILE ], SECTIONS):
    print('Preprocessing %s...' % fn)
    h = open(fn, 'wt')

    for section in range(sections[0], sections[1]+1):
        fileids = [ i for i in ptb.fileids() if i.startswith(str(section).zfill(2)) ]

        for sent, tree in zip(ptb.sents(fileids), ptb.parsed_sents(fileids)):
            sent = [ normalize(word) if normalize(word) in vocab else '<unk>' for word in sent ]
            lin = linearize(tree)

            h.write('%s\t%s\n' % (' '.join(sent), lin))

    h.close()

    print('Done.')
