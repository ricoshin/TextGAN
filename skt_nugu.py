import os

from konlpy.tag import Twitter
import numpy as np
from tqdm import tqdm


def load_skt_nugu(data_dir, maximum):
    vocab = set()

    def parse_file(lines, vocab):
        twitter = Twitter()
        data = dict()
        for i, line in tqdm(enumerate(lines), desc=lines.name,
                            mininterval=0.5):
            line = line.strip().split('\t')
            if len(line) is not 2:
                continue
            key, value = line
            if data.get(key) is None:
                data[key] = []
            if len(data[key]) == maximum:
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


def load_skt_nugu_samples(data_dir):
    answers = []
    questions = []
    vocab = set()
    twitter = Twitter()

    with open(os.path.join(data_dir, 'question_samples.txt')) as lines:
        for line in tqdm(lines, desc=lines.name, mininterval=0.5):
            key, value = line.strip().split('\t')

            tokens = twitter.morphs(value)
            questions.append(tokens)

            answers.append(key)

            vocab.update(tokens)
    return questions, answers, vocab


def load_category_line_nums(data_dir, categories):
    def parse_file(lines):
        line_nums = dict()
        for num, line in enumerate(tqdm(lines, desc=lines.name,
                                        mininterval=0.5)):
            line = line.strip().split('\t')
            if len(line) is not 2:
                continue
            category = line[0].split('_')[0].split('.')[0]
            if category not in categories:
                continue
            if line_nums.get(category) is None:
                line_nums[category] = []
            line_nums[category].append(num)
        return line_nums

    with open(os.path.join(data_dir, 'questions.txt')) as lines:
        ques_line_nums = parse_file(lines)

    return ques_line_nums


def _make_samples(data_dir, categories, num_samples):
    data_dir = '/v2/data_center/skt-nugu'
    categories = ['car', 'hair', 'fashion', 'personal', 'emotion', 'aloners',
                  'praise', 'etc', 'diet', 'date']

    line_nums = load_category_line_nums(data_dir, categories)

    sampled_line_nums = list()
    num_samples = 10000
    for key, line_nums in line_nums.items():
        sampled_line_nums.extend(
            np.random.permutation(line_nums)[:num_samples])

    samples = _load_question_samples(data_dir, sampled_line_nums)
    _save_samples(data_dir, samples)
    return samples


def _save_samples(data_dir, samples):
    fname = os.path.join(data_dir, 'question_samples.txt')
    with open(fname, 'w') as f:
        for key, value in tqdm(samples.items(), desc=fname):
            for line in tqdm(value, desc=key):
                f.write('{key}\t{value}\n'.format(key=key, value=line))


def _load_question_samples(data_dir, line_nums):
    line_nums = sorted(line_nums)
    samples = dict()

    with open(os.path.join(data_dir, 'questions.txt')) as lines:
        for num, line in enumerate(tqdm(lines, desc=lines.name,
                                        mininterval=0.5)):
            if not line_nums:
                break
            if num != line_nums[0]:
                continue
            line_nums = line_nums[1:]

            key, value = line.strip().split('\t')
            category = key.split('_')[0].split('.')[0]
            if samples.get(category) is None:
                samples[category] = []
            samples[category].append(value)

    return samples
