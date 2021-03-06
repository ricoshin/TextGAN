import numpy as np
import random


class Dataset(object):
    def __init__(self, data, batch_size, num_vocab, pad_idx, shuffle=True):
        self.que = data[0]
        self.ans = data[1]
        self.batch_size = batch_size
        self.length = len(self.que)
        self.batch_idx = 0
        self.shuffle = True
        self.current_batch_que = None
        self.current_batch_ans = None
        self.pad_idx = pad_idx
        self.num_vocab = num_vocab
        self.__random_shuffle()

    def __random_shuffle(self):
        idx = np.random.permutation(self.length)
        self.que = self.que[idx]
        self.ans = self.ans[idx]

    def __dense_to_onehot_batch(self, batch):
        return (np.arange(self.num_vocab) == batch[:, :, None]).astype(int)

    def set_next_batch(self):
        under = self.batch_idx
        upper = under + self.batch_size
        max_ = self.length
        if upper <= max_:
            batch_que = self.que[under:upper]
            batch_ans = self.ans[under:upper]
            under = upper
        else:
            rest = upper - max_
            if self.shuffle is True:
                self.__random_shuffle()
            batch_que = np.concatenate(
                            (self.que[under:max_], self.que[0:rest]))
            batch_ans = np.concatenate(
                            (self.ans[under:max_], self.ans[0:rest]))
            under = rest
        self.current_batch_que = self.__dense_to_onehot_batch(batch_que)
        self.current_batch_ans = batch_ans
        self.batch_idx = under


class BatchGenerator(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def swap_random_words(self, batch, pad_idx):
        import copy
        batch_clone = copy.deepcopy(batch)  # for preserving raw data
        for sent in batch_clone:
            if pad_idx in sent:
                len_sent = sent.tolist().index(pad_idx)
            else:  # if there's no PAD at all
                len_sent = len(sent)
            if len_sent < 2:  # if sent is consist of less than 2 words
                continue      # skip over to the next batch
            else:  # prevent duplication
                i, j = random.sample(range(0, len_sent), 2)
            sent[i], sent[j] = sent[j], sent[i]
        return batch_clone

    def dense_to_onehot(self, labels, num_classes):
        return np.eye(num_classes)[labels]

    def get_d_batch(self):
        """ for discriminator pre-training.
            divide data batch(2n) in half : real data(n) + fake data(n) """
        if not self.dataset.batch_size % 2 == 0:
            raise Exception("[!] batch size must be even.")
        half_size = self.dataset.batch_size//2

        self.dataset.set_next_batch()
        # questions
        batch_que = self.dataset.current_batch_que
        que_real = batch_que[0:half_size]
        que_fake_raw = batch_que[half_size:]
        que_fake = self.swap_random_words(que_fake_raw, self.dataset.pad_idx)
        batch_que = np.concatenate((que_real, que_fake))

        # answers
        batch_ans = self.dataset.current_batch_ans

        # labels
        label_real = self.dense_to_onehot(np.ones(half_size, dtype=np.int), 2)
        label_fake = self.dense_to_onehot(np.zeros(half_size, dtype=np.int), 2)
        batch_label = np.concatenate((label_real, label_fake))

        return batch_que, batch_ans, batch_label

    def get_gan_data_batch(self):
        self.dataset.set_next_batch()
        return self.dataset.current_batch_que, self.dataset.current_batch_ans

    def get_gan_label_batch(self):
        batch_size = self.dataset.batch_size
        label_real = self.dense_to_onehot(np.ones(batch_size, dtype=np.int), 2)
        label_fake = self.dense_to_onehot(
                         np.zeros(batch_size, dtype=np.int), 2)
        return np.concatenate((label_real, label_fake))

    def get_binary_label_batch(self, is_true):
        batch_size = self.dataset.batch_size
        if is_true:
            label = self.dense_to_onehot(np.ones(batch_size, dtype=np.int), 2)
        else:
            label = self.dense_to_onehot(np.zeros(batch_size, dtype=np.int), 2)
        return label
