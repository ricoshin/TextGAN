import os

from konlpy.tag import Twitter
from tqdm import tqdm


def load_skt_nugu(data_dir):
    vocab = set()

    def parse_file(lines, vocab):
        twitter = Twitter()
        data = dict()
        for i, line in tqdm(enumerate(lines), desc=lines.name, mininterval=0.5):
            line = line.strip().split('\t')
            if len(line) is not 2:
                continue
            key, value = line
            if data.get(key) is None:
                data[key] = []
            if len(data[key]) == 100:
                continue
            tokens = twitter.morphs(value)
            data[key].append(tokens)
            vocab.update(tokens)
        return data

    with open(os.path.join(data_dir, 'questions.txt')) as lines:
        questions = parse_file(lines, vocab)

    with open(os.path.join(data_dir, 'answers.txt')) as lines:
        answers = parse_file(lines, vocab)

    return questions, answers, vocab
