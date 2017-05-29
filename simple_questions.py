import os

from nltk.tokenize import word_tokenize


def load_simple_questions(config, lower=True):
    """
    Returns:
        (train, valid, vocab)
        train: training set (questions, answers)
        valid: validation set (questions, answers)
        vocab: set(str)
    """
    data_dir = os.path.join(config.data_dir, 'SimpleQuestions')
    vocab = set()

    def parse_file(lines, vocab):
        questions = []
        answers = []
        for line in lines:
            if lower:
                line = line.rstrip().lower()
            else:
                line = line.rstrip()
            ques, ans = line.split('\t')

            ans_tokens = word_tokenize(ans)
            answers.append(ans_tokens)

            ques = ques[2:]  # remove heading number and space
            ques_tokens = word_tokenize(ques)
            questions.append(ques_tokens)

            vocab.update(ques_tokens + ans_tokens)
        return questions, answers

    with open(os.path.join(data_dir, 'train.txt')) as lines:
        train = parse_file(lines, vocab)

    with open(os.path.join(data_dir, 'valid.txt')) as lines:
        valid = parse_file(lines, vocab)

    return train, valid, vocab
