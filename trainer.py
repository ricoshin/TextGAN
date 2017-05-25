from __future__ import print_function

import os
import numpy as np
from tqdm import trange
from collections import deque
from utils import *
import tensorflow as tf
from models import *
from data import TOK_PAD
import random

class Trainer(object):
    def __init__(self, config, train_data, valid_data, W_e_init, word2idx):
        #  self, W_e_init, max_sentence_len, num_classes, vocab_size,
        #  embedding_size, filter_sizes, num_filters, data_format, l2_reg_lambda=0.0
        #MAX_SENTENCE_LEN = 20
        #VOCAB_SIZE = 1000
        #EMBEDDING_SIZE = 300
        self.batch_size = config.batch_size
        self.train_data = np.asarray(train_data)
        self.valid_data = np.asarray(valid_data)
        #self.data = np.random.randint(VOCAB_SIZE, size=(self.batch_size, MAX_SENTENCE_LEN))
        self.W_e_init = np.asarray(W_e_init)
        #self.W_e_init = np.random.rand(VOCAB_SIZE, EMBEDDING_SIZE)
        self.word2idx = word2idx
        self.pad_idx = word2idx[TOK_PAD]

        if self.train_data.shape[1] == self.valid_data.shape[1]:
            self.max_sentence_len = self.train_data.shape[1]
        else:
            raise Exception("[!] max length of sentence in train & valid data MUST be the same")

        self.vocab_size = self.W_e_init.shape[0]
        self.embedding_size = self.W_e_init.shape[1]

        self.filter_sizes = [3,4,5]
        self.num_filters = 300
        self.dropout_keep_prob = 0.5
        self.l2_reg_lambda = 0.0
        self.data_format = config.data_format

        self.optimizer = config.optimizer
        self.z_num = config.z_num

        self.model_dir = config.model_dir # e.g. logs/MyDataSet_xxx
        self.load_path = config.load_path # e.g. logs/MyDataset_xxx(test) or None(train)

        self.use_gpu = config.use_gpu
        self.g_lr, self.d_lr = config.g_lr, config.d_lr

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.valid_step = config.valid_step
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.validation = config.validation

        self.build_model()

        self.saver = tf.train.Saver()
        self.train_summary_writer = tf.summary.FileWriter(self.model_dir+'/train')
        self.valid_summary_writer = tf.summary.FileWriter(self.model_dir+'/valid')
        self.sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300, # 5 minutes
                                global_step=self.global_step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = self.sv.prepare_or_wait_for_session(config=sess_config)


    def build_model(self):
        #  self, W_e_init, max_sentence_len, num_classes, vocab_size,
        #  embedding_size, filter_sizes, num_filters, data_format, l2_reg_lambda=0.0
        self.D = Discriminator(W_e_init=self.W_e_init,
                                max_sentence_len=self.max_sentence_len,
                                num_classes=2,
                                vocab_size=self.vocab_size,
                                embedding_size=self.embedding_size,
                                filter_sizes=self.filter_sizes,
                                num_filters=self.num_filters,
                                data_format=self.data_format,
                                l2_reg_lambda=self.l2_reg_lambda)

        if self.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Not supported optimizer")

        #self.g_lr = tf.Variable(g_lr, name='g_lr')
        self.d_lr = tf.Variable(self.d_lr, name='d_lr')
        #self.g_lr_update = tf.assign(self.g_lr, (self.g_lr * 0.5), name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, (self.d_lr * 0.5), name='d_lr_update')
        #g_optimizer = self.optimizer(self.g_lr)
        d_optimizer = self.optimizer(self.d_lr)

        #g_optim = g_optimizer.minimize(self.G_loss, global_step=self.step, var_list=self.G_var)
        grads_and_vars = d_optimizer.compute_gradients(self.D.loss)
        self.d_train_op = d_optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        # global_step: Optional Variable to increment by one after the variables have been updated.

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for _grads, _vars in grads_and_vars:
            if _grads is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(_vars.name), _grads)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(_vars.name), tf.nn.zero_fraction(_grads))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        self.summary_op = tf.summary.merge([
            tf.summary.scalar("D_loss", self.D.loss),
            tf.summary.scalar("D_accuracy", self.D.accuracy),
            tf.summary.scalar("D_learning_rate", self.d_lr),
            grad_summaries_merged
        ])

        self.d_lr_update = tf.assign(self.d_lr, (self.d_lr * 0.5), name='d_lr_update')

    def train(self):

        class batch_generator(object):
            def __init__(self, batch_size, train_data, valid_data=None, shuffle=True):
                self.batch_size = batch_size
                self.shuffle = shuffle

                self.train_data = train_data
                self.train_len = len(train_data)
                self.train_idx = 0

                self.valid_data = valid_data
                if valid_data:
                    self.valid_len = len(valid_data)
                    self.valid_idx = 0

            def get_batch(data, under, max):
                upper = under + batch_size
                if upper <= max:
                    batch = data[under:upper]
                    under = upper
                else:
                    rest = upper - max
                    if self.shuffle is True:
                        np.random.shuffle(data)
                    batch = np.concatenate((data[under:max], data[0:rest]))
                    under = rest
                return batch, under


            def swap_random_words(batch, pad_idx):
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

            def convert_to_onehot(labels, num_classes):
                return np.eye(num_classes)[labels]

            def generate_true_fake_data_batch(batch, batch_size, pad_idx):
                data_real = batch[0:int(self.batch_size/2)]
                data_fake_raw = batch[int(self.batch_size/2):]
                data_fake = swap_random_words(data_fake_raw, pad_idx)
                data_concatenated = np.concatenate((data_real, data_fake))
                return data_concatenated

            def generate_true_fake_label_batch(batch_size):
                label_real = convert_to_onehot(np.ones(int(batch_size/2), dtype=np.int), 2)
                label_fake = convert_to_onehot(np.zeros(batch_size-int(batch_size/2), dtype=np.int), 2)
                label_concatenated = np.concatenate((label_real, label_fake))
                return label_concatenated

            def shuffle_bath():
                train_data = generate_true_fake_data_batch(batch, self.batch_size, self.pad_idx)


            def batch_train_valid():
                self.train_batch, self.train_idx = get_batch(self.train_data, self.train_idx, self.train_len)
                if self.valid_data:
                    self.valid_batch, self.valid_idx = get_batch(self.valid_data, self.valid_idx, self.valid_len)


            if self.validation is True:
                # generate valid data & labels
                print('Generating data & labels for validation monitoring..')
                valid_data = generate_true_fake_data_batch(self.valid_data, len(self.valid_data), self.pad_idx)
                valid_label = generate_true_fake_label_batch(len(self.valid_data))
                print('Done.')

        for step, batch in shuffle_batch(self.train_data, self.batch_size, self.max_step):

            train_data = generate_true_fake_data_batch(batch, self.batch_size, self.pad_idx)
            train_label = generate_true_fake_label_batch(self.batch_size)

            fetch_dict = {
                "d_train_op": self.d_train_op,
                "global_step": self.global_step,
                "d_train_loss": self.D.loss,
                "d_train_accuracy": self.D.accuracy
            }
            feed_dict = {
                self.D.input_x: train_data,
                self.D.input_y: train_label,
                self.D.dropout_keep_prob: self.dropout_keep_prob
            }


            if step % self.log_step == 0: # default : 50
                fetch_dict.update({
                    "summary": self.summary_op
                })
            result = self.sess.run(fetch_dict, feed_dict)

            if  step% self.log_step == 0: # default : 50
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                d_loss = result['d_train_loss']
                d_accuracy = result['d_train_accuracy']

                print("[{}/{}] D_loss: {:.6f} D_accuracy: {:.6f} ". \
                      format(step, self.max_step, d_loss, d_accuracy))

            if self.validation is True:
                if step % self.valid_step == 0:

                  fetch_dict = {
                  "d_valid_loss": self.D.loss,
                  "d_valid_accuracy": self.D.accuracy,
                  "summary" : self.summary_op
                  }

                  feed_dict = {
                  self.D.input_x: valid_data,
                  self.D.input_y: valid_label,
                  self.D.dropout_keep_prob: 1 # keep all units ON
                  }

                  result = self.sess.run(fetch_dict, feed_dict)
                  d_loss = result['d_valid_loss']
                  d_accuracy = result['d_valid_accuracy']
                  print("[Validation] D_loss: {:.6f} D_accuracy: {:.6f} ". \
                        format(d_loss, d_accuracy))
                  self.summary_writer.add_summary(result['summary'], step)
                  self.summary_writer.flush()

            if step % (self.log_step * 10) == 0: # default : every 100 steps
                pass

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run(self.d_lr_update)

    def test(self):

        pass
