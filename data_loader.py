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

class Batch_generator(object):

    def __init__(self, train_dataset, valid_dataset=None):
        self.train_dataset = train_dataset
        if valid_dataset:
            self.validation = True
            self.valid_dataset = valid_dataset

    def __set_current_batch(self, dataset):
        data = dataset.data
        under = dataset.batch_idx
        upper = under + dataset.batch_size
        max = dataset.length
        if upper <= max:
            batch = data[under:upper]
            under = upper
        else:
            rest = upper - max
            if dataset.shuffle is True:
                np.random.shuffle(data)
            batch = np.concatenate((data[under:max], data[0:rest]))
            under = rest
        dataset.current_batch = batch
        dataset.batch_idx = under

    def __swap_random_words(self, batch, pad_idx):
        import copy
        batch_clone = copy.deepcopy(batch) # for preserving raw data
        for sent in batch_clone:
            try:
                if pad_idx in sent:
                    len_sent = sent.tolist().index(pad_idx)
                else: # if there's no PAD at all
                    len_sent = len(sent)
                if len_sent < 2: # if sent is consist of less than 2 words
                    continue     # skip over to the next batch
                else: # prevent duplication
                    i,j = random.sample(range(0,len_sent), 2)
            except:
                import pdb; pdb.set_trace()
            sent[i], sent[j] = sent[j], sent[i]
        return batch_clone

    def __dense_to_onehot(self, labels, num_classes):
        return np.eye(num_classes)[labels]

    def __generate_true_fake_data_batch(self, dataset):
        half_size = int(dataset.batch_size/2)
        data_real = dataset.current_batch[0:half_size]
        data_fake_raw = dataset.current_batch[half_size:]
        data_fake = self.__swap_random_words(data_fake_raw, dataset.pad_idx) # Swap!
        data_concatenated = np.concatenate((data_real, data_fake))
        return data_concatenated

    def __generate_true_fake_label_batch(self, dataset):
        half_size = int(dataset.batch_size/2)
        label_real = self.__dense_to_onehot(np.ones(half_size, dtype=np.int), 2)
        label_fake = self.__dense_to_onehot(np.zeros(dataset.batch_size-half_size, dtype=np.int), 2)
        label_concatenated = np.concatenate((label_real, label_fake))
        return label_concatenated

    def get_train_batch(self):
        self.__set_current_batch(self.train_dataset)
        train_data = self.__generate_true_fake_data_batch(self.train_dataset)
        train_label = self.__generate_true_fake_label_batch(self.train_dataset)
        return train_data, train_label

    def get_valid_batch(self):
        if self.validation:
            self.__set_current_batch(self.valid_dataset)
            valid_data = self.__generate_true_fake_data_batch(self.valid_dataset)
            valid_label = self.__generate_true_fake_label_batch(self.valid_dataset)
            return valid_data, valid_label
        else:
            raise Exception("[!] Validation dataset hasn't been initialized.")
