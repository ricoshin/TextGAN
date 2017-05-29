import copy
import csv
import numpy as np
import os
import re
import sys

from progress.bar import Bar
from simple_questions import load_simple_questions
from skt_nugu import load_skt_nugu, load_skt_nugu_samples
from wordvec import (load_fasttext_vocab, load_fasttext_embeddings,
                     load_glove_vocab, load_glove_embeddings)

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
        if examples is None:
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


def convert_to_token(sents, word2idx, trim=True):
    idx2word = {v: k for k, v in word2idx.items()}
    # TODO: it is not trimming but removing only PAD.
    return [[idx2word[idx] for idx in sent
             if not (trim and idx2word[idx] == TOK_PAD)] for sent in sents]


def remove_unknown_answers(data, vocab):
    data = zip(data[0], data[1])

    questions = []
    answers = []
    new_vocab = set()
    for q, a in data:
        if len(a) == 1 and a[0] in vocab:
            questions.append(q)
            answers.append(a)
        elif len(a) > 1 and '_'.join(a) in vocab:
            questions.append(q)
            a = '_'.join(a)
            answers.append([a])
            new_vocab.add(a)
    return (questions, answers), new_vocab


def load_skt_nugu_sample_dataset(config):
    data_npz = os.path.join(config.data_dir, 'data_skt_nugu_sample.npz')
    word2idx_txt = os.path.join(config.data_dir, 'word2idx_skt_nugu_sample.txt')
    ans2idx_txt = os.path.join(config.data_dir, 'ans2idx_skt_nugu_sample.txt')

    if (os.path.exists(data_npz) and os.path.exists(word2idx_txt) and
            os.path.exists(ans2idx_txt)):
        npz = np.load(data_npz)
        embd_mat = npz['embd_mat']
        ques = npz['ques'].astype(np.int32)
        ans = npz['ans'].astype(np.int32)

        with open(word2idx_txt) as f:
            reader = csv.reader(f, delimiter='\t')
            word2idx = {row[0]: int(row[1]) for row in reader}
        with open(ans2idx_txt) as f:
            reader = csv.reader(f, delimiter='\t')
            ans2idx = {row[0]: int(row[1]) for row in reader}

        train = ques, ans
        return train, ans2idx, embd_mat, word2idx

    path = os.path.join(config.data_dir, 'fasttext')
    fast_vocab = load_fasttext_vocab(path, 'ko')

    path = os.path.join(config.data_dir, 'skt-nugu')
    ques, ans, nugu_vocab = load_skt_nugu_samples(path)

    import ipdb; ipdb.set_trace()

    unknown_vocab = nugu_vocab-fast_vocab
    ques = replace_unknowns(ques, unknown_vocab)

    vocab = nugu_vocab-unknown_vocab
    vocab.update([TOK_UNK, TOK_PAD])

    max_ques_len = max(len(sent) for sent in ques)
    ques = append_pads(ques, max_ques_len)

    embd_mat, word2idx = load_fasttext_embeddings(os.path.join(config.data_dir,
                                                               'fasttext'),
                                                  'ko', vocab)

    ques = convert_to_idx(ques, word2idx)
    ans2idx = {ans: i for i, ans in enumerate(set(ans))}
    ans = [ans2idx[a] for a in ans]

    with open(word2idx_txt, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(word2idx.items())
    with open(ans2idx_txt, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(ans2idx.items())
    np.savez(data_npz, embd_mat=embd_mat, ques=ques, ans=ans)

    train = np.array(ques), np.array(ans)
    return train, ans2idx, embd_mat, word2idx


def load_nugu_dataset(config):
    data_npz = os.path.join(config.data_dir, 'data_skt_nugu.npz')
    word2idx_txt = os.path.join(config.data_dir, 'word2idx_skt_nugu.txt')

    path = os.path.join(config.data_dir, 'fasttext')
    fast_vocab = load_fasttext_vocab(path, 'ko')

    path = os.path.join(config.data_dir, 'skt-nugu')
    ques, ans, nugu_vocab = load_skt_nugu(path, 200)

    ques_keys = ques.keys()
    ans_keys = ans.keys()
    for key in set(ques_keys)-set(ans_keys):
        del ques[key]
    for key in set(ans_keys)-set(ques_keys):
        del ans[key]
    assert(set(ques.keys()) == set(ans.keys()))
    keys = set(ans.keys())

    pairs = []
    for key in keys:
        for a in ans[key]:
            for q in ques[key]:
                pairs.append((q, a))
    ques = [pair[0] for pair in pairs]
    ans = [pair[1] for pair in pairs]

    unknown_vocab = nugu_vocab-fast_vocab
    ques = replace_unknowns(ques, unknown_vocab)
    ans = replace_unknowns(ans, unknown_vocab)
    vocab = nugu_vocab-unknown_vocab
    vocab.update([TOK_UNK, TOK_PAD])

    max_ques_len = max(len(sent) for sent in ques)
    ques = append_pads(ques, max_ques_len)
    max_ans_len = max(len(sent) for sent in ans)
    ans = append_pads(ans, max_ans_len)

    embd_mat, word2idx = load_fasttext_embeddings(os.path.join(config.data_dir,
                                                               'fasttext'),
                                                  'ko', vocab)

    ques = convert_to_idx(ques, word2idx)
    ans = convert_to_idx(ans, word2idx)

    with open(word2idx_txt, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(word2idx.items())
    np.savez(data_npz, embd_mat=embd_mat, ques=ques, ans=ans)

    return ques, ans, embd_mat, word2idx


def load_simple_questions_dataset(config, force_reload=False):
    bar = Bar(suffix='%(index)d/%(max)d - %(elapsed)ds')

    data_npz = os.path.join(config.data_dir, 'data.npz')
    word2idx_txt = os.path.join(config.data_dir, 'word2idx.txt')

    if (os.path.exists(data_npz) and os.path.exists(word2idx_txt) and
            not force_reload):
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

    bar.max = 8

    bar.message = 'Loading GloVe vocab'
    bar.next()
    glove_vocab = load_glove_vocab(os.path.join(config.data_dir, 'glove'),
                                   '42B', 300)

    bar.message = 'Loading SimpleQuestions'
    bar.next()
    train, valid, dataset_vocab = load_simple_questions(config)

    bar.message = 'Removing unknown answers'
    bar.next()
    train, new_vocab = remove_unknown_answers(train, glove_vocab)
    dataset_vocab.update(new_vocab)

    valid, new_vocab = remove_unknown_answers(valid, glove_vocab)
    dataset_vocab.update(new_vocab)

    train_q, train_a = train[0], train[1]
    valid_q, valid_a = valid[0], valid[1]

    bar.message = 'Replacing unknown tokens'
    bar.next()
    unknowns = dataset_vocab-glove_vocab
    train_q = replace_unknowns(train_q, unknowns)
    train_a = replace_unknowns(train_a, unknowns)
    valid_q = replace_unknowns(valid_q, unknowns)
    valid_a = replace_unknowns(valid_a, unknowns)
    vocab = dataset_vocab-unknowns

    bar.message = 'Appending pads'
    bar.next()
    max_len = max(len(sent) for sent in train_q+valid_q)
    train_q = append_pads(train_q, max_len)
    valid_q = append_pads(valid_q, max_len)
    vocab.update([TOK_UNK, TOK_PAD])

    bar.message = 'Loading GloVe embeddings'
    bar.next()
    embd_mat, word2idx = load_glove_embeddings(os.path.join(config.data_dir,
                                                            'glove'),
                                               '42B', 300, vocab)

    bar.message = 'Converting token to index'
    bar.next()
    train_q = convert_to_idx(train_q, word2idx)
    train_a = convert_to_idx(train_a, word2idx)
    valid_q = convert_to_idx(valid_q, word2idx)
    valid_a = convert_to_idx(valid_a, word2idx)

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
    train = np.array(train_q), np.array(train_a)
    valid = np.array(valid_q), np.array(valid_a)
    return train, valid, embd_mat, word2idx
