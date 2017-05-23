from nltk.tokenize import word_tokenize


def load_simple_questions(file_path, lower):
    """
    Returns:
        questions: list(list(str))
        vocab: set(str)
    """
    questions = []
    answers =  []
    vocab = set()
    with open(file_path) as lines:
        for line in lines:
            if lower:
                line = line.rstrip().lower()
            else:
                line = line.rstrip()
            ques, ans = line.split('\t')
            ques = ques[2:] # remove heading number and space
            ques_tokens = word_tokenize(ques)
            questions.append(ques_tokens)
            vocab.update(ques_tokens)
    return questions, answers, vocab
