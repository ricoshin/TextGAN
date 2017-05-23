import copy
import csv
import numpy as np
import os
import re

from glove import load_glove_vocab, load_glove_embeddings
from simple_questions import load_simple_questions


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

def append_pads(sents):
    """
    Args:
        sents: list(list(str))
    Returns:
        list(list(str))
    """
    sents = copy.deepcopy(sents)
    max_len = max(len(sent) for sent in sents)
    for sent in sents:
        num_pads = max_len-len(sent)
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
    ret = []
    for sent in sents:
        try:
            ret.append([word2idx[token] for token in sent])
        except:
            import ipdb; ipdb.set_trace()
    return ret
    #return [[word2idx[token] for token in sent] for sent in sents]

def load_simple_questions_dataset():
    if os.path.exists('data/data.npz') and os.path.exists('data/word2idx.txt'):
        print('Load saved data')
        npz = np.load('data/data.npz')
        embd_mat = npz['embd_mat']
        questions = npz['questions'].astype(np.int32)
        with open('data/word2idx.txt') as f:
            reader = csv.reader(f, delimiter='\t')
            word2idx = {row[0]: int(row[1]) for row in reader}
        return questions, embd_mat, word2idx
    print('Load SimpleQuestions')
    questions, answers, sq_vocab = load_simple_questions('data/SimpleQuestions/train.txt', lower=True)

    print('Load GloVe vocab')
    glove_vocab = load_glove_vocab('data/glove', '42B', 300)

    print('Replace unknown tokens')
    unknowns = sq_vocab-glove_vocab
    questions = replace_unknowns(questions, unknowns)
    vocab = sq_vocab-unknowns

    print('Append pads')
    max_question_len = max(len(question) for question in questions)
    questions = append_pads(questions)
    vocab.update([TOK_UNK, TOK_PAD])

    print('Load GloVe embeddings')
    embd_mat, word2idx = load_glove_embeddings('data/glove', '42B', 300, vocab)

    print('Convert to index')
    questions = convert_to_idx(questions, word2idx)

    print('Save data')
    with open('data/word2idx.txt', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(word2idx.items())
    np.savez('data/data.npz', embd_mat=embd_mat, questions=questions)

    return questions, embd_mat, word2idx

def get_batch(data, batch_size):
    for offset in range(0, len(data), batch_size):
        yield data[offset:offset+batch_size]
