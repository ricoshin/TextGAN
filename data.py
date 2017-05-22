from collections import Counter
from glob import glob
import os
import re

import nltk
from nltk.corpus import LazyCorpusLoader
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
import numpy as np


class CorpusReader(PlaintextCorpusReader):
    def __init__(self,
                 root,
                 fields,
                 word_tokenizer=TreebankWordTokenizer(),
                 sent_tokenizer=PunktSentenceTokenizer(),
                 **kwargs):
        known_abbrs = ['mr', 'mrs', 'dr', 'st']
        sent_tokenizer._params.abbrev_types.update(known_abbrs)
        super(CorpusReader, self).__init__(root,
                                           fields,
                                           word_tokenizer=word_tokenizer,
                                           sent_tokenizer=sent_tokenizer,
                                           **kwargs)
def find_abbr_candidates(text):
    abbr_regex = re.compile('(\s\w+\s([A-Z][a-z]*\.([A-Za-z]\.)*)\s\w+)')
    known_abbrs = ['mr.', 'mrs.', 'dr.', 'st.']
    candidates = [x for x in abbr_regex.findall(text)
                  if x[1].lower() not in known_abbrs]
    cand_dict = dict()
    for cand in candidates:
        examples = cand_dict.get(cand[1].lower())
        if examples == None:
            cand_dict[cand[1].lower()] = []
            examples = cand_dict[cand[1].lower()]
        examples.append(cand[0])
    return cand_dict

def load_gutenberg_dataset():
    nltk.download('gutenberg')
    gutenberg = LazyCorpusLoader('gutenberg',
                                 CorpusReader,
                                 r'(?!\.).*\.txt',
                                 encoding='latin1')
    sents = [[word.lower() for word in sent] for sent in gutenberg.sents()]
    vocab = Counter()
    for sent in sents:
        vocab.update(sent)
    return sents, vocab

def load_glove_with_vocab(data_dir, size, vocab):
    """
    Args:
        data_dir: Path to pre-trained GloVe embeddings directory
        size: Word embedding size: 50, 100, 200, or 300
    Returns:
        Tuple of GloVe word embeddings matrix of shape [len(vocab), size] and
        word to index dictionary
    """
    word2embd = dict()
    fname = 'glove.6B.%dd.txt' % size
    with open(os.path.join(data_dir, fname)) as f:
        for line in f:
            l = line.split()
            word = l[0]
            if word not in vocab:
                continue
            embd = [float(x) for x in l[1:]]
            word2embd[word] = embd

    word2idx = {word: i for i, word in enumerate(vocab)}
    embd_mat = np.ndarray([len(vocab), size], dtype=np.float32)
    for word in vocab:
        embd_mat[word2idx[word]] = word2embd.get(word, np.random.random(size))

    import ipdb; ipdb.set_trace()

    assert(len(embd_mat) == len(word2idx))
    return np.array(embd_mat), word2idx

def load_glove_words(data_dir, num_tokens, size):
    sizes = {
        '6B': [50, 100, 200, 300],
        '42B': [300],
    }
    assert(num_tokens in sizes.keys())
    assert(size in sizes[num_tokens])

    fname = 'glove.{token}.{size}d.txt'.format(token=num_tokens, size=size)
    vocab = []
    with open(os.path.join(data_dir, fname)) as f:
        vocab = [line.split()[0] for line in f]
    return vocab

def filter_gutenberg(vocab):
    gutenberg = LazyCorpusLoader('gutenberg',
                                 CorpusReader,
                                 r'(?!\.).*\.txt',
                                 encoding='latin1')
    gb_vocab = set()
    for sent in gutenberg.sents():
        gb_vocab.update(list(map(str.lower, sent)))
    vocab = gb_vocab.intersection(vocab)
    sents = []
    for sent in gutenberg.sents():
        for word in sent:
            if word not in vocab:
                break
        else:
            sents.append(sent)
    return sents

vocab = load_glove_words('data/glove', '42B', 300)
print('vocab loaded')
sents = filter_gutenberg(vocab)
print('sents loaded')
#sents, vocab = load_gutenberg_dataset()
#word_embds, word2idx = load_glove('data/glove', 50, set(vocab.keys()))
