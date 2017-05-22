import os
import re


file_names = {
    ("qa1_single-supporting-fact_train.txt", "qa1_single-supporting-fact_test.txt"),
    ("qa2_two-supporting-facts_train.txt", "qa2_two-supporting-facts_test.txt"),
    ("qa3_three-supporting-facts_train.txt", "qa3_three-supporting-facts_test.txt"),
    ("qa4_two-arg-relations_train.txt", "qa4_two-arg-relations_test.txt"),
    ("qa5_three-arg-relations_train.txt", "qa5_three-arg-relations_test.txt"),
    ("qa6_yes-no-questions_train.txt", "qa6_yes-no-questions_test.txt"),
    ("qa7_counting_train.txt", "qa7_counting_test.txt"),
    ("qa8_lists-sets_train.txt", "qa8_lists-sets_test.txt"),
    ("qa9_simple-negation_train.txt", "qa9_simple-negation_test.txt"),
    ("qa10_indefinite-knowledge_train.txt", "qa10_indefinite-knowledge_test.txt"),
    ("qa11_basic-coreference_train.txt", "qa11_basic-coreference_test.txt"),
    ("qa12_conjunction_train.txt", "qa12_conjunction_test.txt"),
    ("qa13_compound-coreference_train.txt", "qa13_compound-coreference_test.txt"),
    ("qa14_time-reasoning_train.txt", "qa14_time-reasoning_test.txt"),
    ("qa15_basic-deduction_train.txt", "qa15_basic-deduction_test.txt"),
    ("qa16_basic-induction_train.txt", "qa16_basic-induction_test.txt"),
    ("qa17_positional-reasoning_train.txt", "qa17_positional-reasoning_test.txt"),
    ("qa18_size-reasoning_train.txt", "qa18_size-reasoning_test.txt"),
    ("qa19_path-finding_train.txt", "qa19_path-finding_test.txt"),
    ("qa20_agents-motivations_train.txt", "qa20_agents-motivations_test.txt"),
}

file_names = {i+1: (train, test) for i, (train, test) in enumerate(file_names)}

def tokenize(s):
    return [x for x in re.split('(\W+)', s)
            if x.strip() and x.strip() != '.' and x.strip() != '?']

def parse_task_file(f, vocab):
    """
    Args:
        f: Task file
        vocab: Vocabulary set to update with words found in task file f
    Returns:
        List of tuple of context(multiple sentences), question(one sentence),
        and answer(one word)
    """
    data = []
    for line in f:
        line = line.strip().lower()
        i = line.index(' ')
        n, line = int(line[:i]), line[i+1:]
        if n == 1:
            story = []
        if '\t' in line:
            q, a, _ = line.split('\t')
            q = tokenize(q)
            vocab.update(q+[a])
            data.append((story, q, a))
        else:
            s = tokenize(line)
            vocab.update(s)
            story.append(s)
    return data

def load_babi_task(data_dir, task_id):
    """
    Args:
        data_dir: Path to babi problem set directory
        task_id: Task ID
    Returns:
        Training and test list of tuple of context, question, and answer, and
        vocabulary set
    """
    train_file, test_file = file_names[task_id]
    train_file = open(os.path.join(data_dir, train_file))
    test_file = open(os.path.join(data_dir, test_file))
    vocab = set()
    train = parse_task_file(train_file, vocab)
    test = parse_task_file(test_file, vocab)
    return train, test, vocab

pairs = []
for i in range(1, 21):
    train, test, vocab = load_babi_task('data/babi/en-10k', i)
    for c, q, a in train+test:
        pairs.append((' '.join(q), a))
questions = set(map(lambda x: x[0], pairs))
answers = set(map(lambda x: x[1], pairs))
