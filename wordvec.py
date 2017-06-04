import os

import numpy as np
from tqdm import tqdm


def _load_vocab(fname):
    """Load vocabulary from `fname`
    `fname` must be in format as:
    word\tfloat_1 float_2 float_3 ... float_n
    Args:
        fname: path to a word vectors file
    Returns:
        a set of words
    """
    with open(fname) as lines:
        vocab = set(line.split()[0] for line in tqdm(lines,
                                                     desc=lines.name,
                                                     mininterval=0.5))
    return vocab


def _load_embeddings(fname, size, vocab, delimiter=' '):
    """Load word vectors corresponding to words in `vocab` from `fname`
    `fname` must be in format as:
    word\tfloat_1 float_2 float_3 ... float_n
    Args:
        fname: path to a word vectors file
        size: word vector dimension
        vocab: a set of words to get word vectors
    """
    word2embd = dict()

    with open(fname) as lines:
        for line in tqdm(lines, desc=lines.name, mininterval=0.5):
            l = line.strip().split(delimiter)
            word = delimiter.join(l[:-size])
            embd = [float(x) for x in l[-size:]]
            word2embd[word] = embd

    word2idx = {word: i for i, word in enumerate(vocab)}
    embd_mat = np.ndarray([len(vocab), size], dtype=np.float32)
    for word in vocab:
        embd_mat[word2idx[word]] = word2embd.get(word, np.random.random(size))

    return np.array(embd_mat), word2idx


def load_fasttext_vocab(data_dir, lang):
    fname = 'wiki.{lang}.vec'.format(lang=lang)
    return _load_vocab(os.path.join(data_dir, fname))


def load_fasttext_embeddings(data_dir, lang, vocab):
    fname = 'wiki.{lang}.vec'.format(lang=lang)
    return _load_embeddings(os.path.join(data_dir, fname), 300, vocab)


_glove_spec = {
    '6B': [50, 100, 200, 300],
    '42B': [300],
}


def load_glove_vocab(data_dir, num_tokens, size):
    assert(num_tokens in _glove_spec.keys())
    assert(size in _glove_spec[num_tokens])

    fname = 'glove.{token}.{size}d.txt'.format(token=num_tokens, size=size)
    return _load_vocab(os.path.join(data_dir, fname))


def load_glove_embeddings(data_dir, num_tokens, size, vocab):
    assert(num_tokens in _glove_spec.keys())
    assert(size in _glove_spec[num_tokens])

    fname = 'glove.{token}.{size}d.txt'.format(token=num_tokens, size=size)
    return _load_embeddings(os.path.join(data_dir, fname), size, vocab)
