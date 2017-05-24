import copy
import csv
import numpy as np
import os
import re

from glove import load_glove_vocab, load_glove_embeddings
from progress.bar import Bar
from simple_questions import load_simple_questions
import sys

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')

TOK_UNK = '-unk-'
TOK_PAD = '-pad-'

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

def replace_unknowns(sents, unknowns):
    """
    Args:
        sents: list(list(str))
        unknowns: set(str)
    Returns:
        list(list(str))
    """
    def replace(sent):
        return [token if token not in unknowns else TOK_UNK for token in sent]
    return list(map(replace, sents))

def append_pads(sents, max_len):
    """
    Args:
        sents: list(list(str))
    Returns:
        list(list(str))
    """
    sents = copy.deepcopy(sents)
    for sent in sents:
        num_pads = max_len-len(sent)
        assert(num_pads >= 0)
        sent.extend([TOK_PAD for _ in range(num_pads)])
    return sents

def convert_to_idx(sents, word2idx):
    """
    Args:
        sents: list(list(str))
        word2idx: dict(str: number)
    Returns:
        list(list(number))
    """
    return [[word2idx[token] for token in sent] for sent in sents]

def load_simple_questions_dataset(config):
    bar = Bar(suffix='%(index)d/%(max)d - %(elapsed)ds')

    data_npz = os.path.join(config.data_dir, 'data.npz')
    word2idx_txt = os.path.join(config.data_dir, 'word2idx.txt')

    if os.path.exists(data_npz) and os.path.exists(word2idx_txt):
        bar.max = 2

        bar.message = 'Loading npz'
        bar.next()
        npz = np.load(data_npz)
        embd_mat = npz['embd_mat']
        train_ques = npz['train_ques'].astype(np.int32)
        train_ans = npz['train_ans'].astype(np.int32)
        valid_ques = npz['valid_ques'].astype(np.int32)
        valid_ans = npz['valid_ans'].astype(np.int32)

        bar.message = 'Loading word2idx'
        bar.next()
        with open(word2idx_txt) as f:
            reader = csv.reader(f, delimiter='\t')
            word2idx = {row[0]: int(row[1]) for row in reader}

        bar.finish()
        train = train_ques, train_ans
        valid = valid_ques, valid_ans
        return train, valid, embd_mat, word2idx

    bar.max = 7

    bar.message = 'Loading SimpleQuestions'
    bar.next()
    train, valid, sq_vocab = load_simple_questions(config)
    train_q, train_a = train[0], train[1]
    valid_q, valid_a= valid[0], valid[1]

    bar.message = 'Loading GloVe vocab'
    bar.next()
    glove_vocab = load_glove_vocab(config, '42B', 300)

    bar.message = 'Replacing unknown tokens'
    bar.next()
    unknowns = sq_vocab-glove_vocab
    train_q = replace_unknowns(train_q, unknowns)
    valid_q= replace_unknowns(valid_q, unknowns)
    vocab = sq_vocab-unknowns

    bar.message = 'Appending pads'
    bar.next()
    max_len = max(len(sent) for sent in train_q+valid_q)
    train_q = append_pads(train_q, max_len)
    valid_q = append_pads(valid_q, max_len)
    vocab.update([TOK_UNK, TOK_PAD])

    bar.message = 'Loading GloVe embeddings'
    bar.next()
    embd_mat, word2idx = load_glove_embeddings(config, '42B', 300, vocab)

    bar.message = 'Converting token to index'
    bar.next()
    train_q = convert_to_idx(train_q, word2idx)
    valid_q = convert_to_idx(valid_q, word2idx)

    bar.message = 'Saving processed data'
    bar.next()
    with open(word2idx_txt, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(word2idx.items())
    data_dict = dict(embd_mat=embd_mat,
                     train_ques=train_q,
                     train_ans=train_a,
                     valid_ques=valid_q,
                     valid_ans=valid_a)
    np.savez(data_npz, **data_dict)

    bar.finish()
    train = train_q, train_a
    valid = valid_q, valid_a
    return train, valid, embd_mat, word2idx
