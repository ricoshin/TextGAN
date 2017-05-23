from nltk.corpus import LazyCorpusLoader
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer


class CorpusReader(PlaintextCorpusReader):
    def __init__(self,
                 root,
                 fields,
                 word_tokenizer=TreebankWordTokenizer(),
                 sent_tokenizer=PunktSentenceTokenizer(),
                 **kwargs):
        known_abbrs = ['mr', 'mrs', 'dr', 'st']
        sent_tokenizer._params.abbrev_types.update(known_abbrs)
        super(CorpusReader, self).__init__(root,
                                           fields,
                                           word_tokenizer=word_tokenizer,
                                           sent_tokenizer=sent_tokenizer,
                                           **kwargs)

def load_gutenberg_dataset():
    nltk.download('gutenberg')
    gutenberg = LazyCorpusLoader('gutenberg',
                                 CorpusReader,
                                 r'(?!\.).*\.txt',
                                 encoding='latin1')
    sents = [[word.lower() for word in sent] for sent in gutenberg.sents()]
    vocab = Counter()
    for sent in sents:
        vocab.update(sent)
    return sents, vocab

def filter_gutenberg(vocab):
    gutenberg = LazyCorpusLoader('gutenberg',
                                 CorpusReader,
                                 r'(?!\.).*\.txt',
                                 encoding='latin1')
    gb_vocab = set()
    for sent in gutenberg.sents():
        gb_vocab.update(list(map(str.lower, sent)))
    vocab = gb_vocab.intersection(vocab)
    sents = []
    for sent in gutenberg.sents():
        for word in sent:
            if word not in vocab:
                break
        else:
            sents.append(sent)
    return sents
