import os

import numpy as np


_sizes = {
    '6B': [50, 100, 200, 300],
    '42B': [300],
}


def load_glove_vocab(config, num_tokens, size):
    assert(num_tokens in _sizes.keys())
    assert(size in _sizes[num_tokens])

    data_dir = os.path.join(config.data_dir, 'glove')
    fname = 'glove.{token}.{size}d.txt'.format(token=num_tokens, size=size)
    with open(os.path.join(data_dir, fname)) as f:
        vocab = set(line.split()[0] for line in f)
    return vocab


def load_glove_embeddings(config, num_tokens, size, vocab):
    """
    Args:
        num_tokens: '6B' or '42B'
        size: Word embedding size: 50, 100, 200, or 300
    Returns:
        Tuple of GloVe word embeddings matrix of shape [len(vocab), size] and
        word to index dictionary
    """
    assert(num_tokens in _sizes.keys())
    assert(size in _sizes[num_tokens])

    word2embd = dict()
    data_dir = os.path.join(config.data_dir, 'glove')
    fname = 'glove.{token}.{size}d.txt'.format(token=num_tokens, size=size)
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

    return np.array(embd_mat), word2idx
