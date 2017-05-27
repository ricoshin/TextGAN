import numpy as np
import random

class Dataset(object):
    def __init__(self, data, batch_size, pad_idx, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.length = len(data)
        self.batch_idx = 0
        self.shuffle = True
        self.current_batch = None
        self.pad_idx = pad_idx

class BatchGenerator(object):

    def __init__(self, dataset, is_pre_train=True):
        self.dataset = dataset


    def __set_next_batch(self):
        data = self.dataset.data
        under = self.dataset.batch_idx
        upper = under + self.dataset.batch_size
        max = self.dataset.length
        if upper <= max:
            batch = data[under:upper]
            under = upper
        else:
            rest = upper - max
            if dataset.shuffle is True:
                np.random.shuffle(data)
            batch = np.concatenate((data[under:max], data[0:rest]))
            under = rest
        self.dataset.current_batch = batch
        self.dataset.batch_idx = under

    def swap_random_words(self, batch, pad_idx):
        import copy
        batch_clone = copy.deepcopy(batch) # for preserving raw data
        for sent in batch_clone:
            if pad_idx in sent:
                len_sent = sent.tolist().index(pad_idx)
            else: # if there's no PAD at all
                len_sent = len(sent)
            if len_sent < 2: # if sent is consist of less than 2 words
                continue     # skip over to the next batch
            else: # prevent duplication
                i,j = random.sample(range(0,len_sent), 2)
            sent[i], sent[j] = sent[j], sent[i]
        return batch_clone

    def dense_to_onehot(self, labels, num_classes):
        return np.eye(num_classes)[labels]

    def get_d_pretrain_data_batch(self):
        """ divide data batch(2n) in half : real data(n) + fake data(n) """
        self.__set_next_batch()
        half_size = int(self.dataset.batch_size/2)
        data_real = self.dataset.current_batch[0:half_size]
        data_fake_raw = self.dataset.current_batch[half_size:]
        data_fake = self.swap_random_words(data_fake_raw, self.dataset.pad_idx) # Swap!
        data_batch = np.concatenate((data_real, data_fake))
        return data_batch

    def get_d_pretrain_label_batch(self):
        """ divide label batch(2n) in half : real label(n) + fake_label(n) """
        half_size = int(dataset.batch_size/2)
        label_real = self.__dense_to_onehot(np.ones(half_size, dtype=np.int), 2)
        label_fake = self.__dense_to_onehot(np.zeros(dataset.batch_size-half_size,
                                                     dtype=np.int), 2)
        label_concatenated = np.concatenate((label_real, label_fake))
        return label_concatenated

    def get_data_batch(self):
        if self.is_pre_train:
            self.__set_current_batch(self.train_dataset)
            train_data = self.generate_d_pretrain_batch(self.train_dataset)
            train_label = self.generate_true_fake_label_batch(self.train_dataset)

        return self.__set_current_batch(self.train_dataset)

    def get_binary_label(self, bool):
        batch_size = self.train_dataset.batch_size
        if label:
            label = self.__dense_to_onehot(np.ones(batch_size, dtype=np.int), 2)
        else:
            label = self.__dense_to_onehot(np.zeros(batch_size, dtype=np.int), 2)
        return label
