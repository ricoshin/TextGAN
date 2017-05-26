from __future__ import print_function

import os
import numpy as np
#from tqdm import trange
from tqdm import tqdm
from collections import deque
from utils import *
import tensorflow as tf
from models import *
from data import TOK_PAD
import random

class Trainer(object):
    def __init__(self, config, train_data, valid_data, W_e_init, word2idx):

        self.batch_size = config.batch_size
        self.train_data = np.asarray(train_data)
        self.valid_data = np.asarray(valid_data)

        self.W_e_init = np.asarray(W_e_init)
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

        self.validation = config.validation
        self.early_stopping = config.early_stopping
        self.early_stopping_metric = config.early_stopping_metric
        self.early_stopping_threshold = config.early_stopping_threshold

        self.build_model()

        self.saver = tf.train.Saver()
        self.train_summary_writer = tf.summary.FileWriter(self.model_dir+'/train')
        self.valid_summary_writer = tf.summary.FileWriter(self.model_dir+'/valid')

        self.sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                #summary_writer=self.summary_writer,
                                save_model_secs=5, # 5 minutes
                                global_step=self.global_step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        #self.sess = self.sv.prepare_or_wait_for_session(config=sess_config)

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
        for _grads, _vars in grads_and_vars:
            if _grads is not None:
                tf.summary.histogram("{}/grad/hist".format(_vars.name), _grads, collections=['train'])
                tf.summary.scalar("{}/grad/sparsity".format(_vars.name),
                                                     tf.nn.zero_fraction(_grads), collections=['train'])

        tf.summary.scalar("D_loss", self.D.loss, collections=['train', 'valid']),
        tf.summary.scalar("D_accuracy", self.D.accuracy, collections=['train', 'valid']),
        tf.summary.scalar("D_learning_rate", self.d_lr, collections=['train']),

        self.train_summary_op = tf.summary.merge_all('train')
        self.valid_summary_op = tf.summary.merge_all('valid')

        self.d_lr_update = tf.assign(self.d_lr, (self.d_lr * 0.5), name='d_lr_update')

    def train_adversarial(self):
        with self.sv.managed_session() as sess:
            pass


    def train(self):
        with self.sv.managed_session() as sess:

            from data_loader import Dataset
            from data_loader import Batch_generator

            train_dataset = Dataset(self.train_data, self.batch_size, self.pad_idx, shuffle=True)
            valid_dataset = Dataset(self.valid_data, self.batch_size, self.pad_idx, shuffle=True)
            batch_generator = Batch_generator(train_dataset, valid_dataset)

            if self.early_stopping:
                valid_metric_history = []

            pbar = tqdm(total = self.max_step)
            step = sess.run(self.global_step)

            if step > 1:
                pbar.update(step)

            while not self.sv.should_stop():

                pbar.update(1)
                train_data, train_label = batch_generator.get_train_batch()

                fetch_dict = {
                    "d_train_op": self.d_train_op,
                    "global_step": self.global_step,
                }
                feed_dict = {
                    self.D.input_x: train_data,
                    self.D.input_y: train_label,
                    self.D.dropout_keep_prob: self.dropout_keep_prob
                }
                result = sess.run(fetch_dict, feed_dict)
                step = result['global_step']


                if  step % self.log_step == 0: # default : 50
                    #self.summary_writer.add_summary(result['summary'], step)
                    fetch_dict.update({
                    "d_train_loss": self.D.loss,
                    "d_train_accuracy": self.D.accuracy,
                    "train_summary": self.train_summary_op
                    })
                    result = sess.run(fetch_dict, feed_dict)
                    d_train_loss = result['d_train_loss']
                    d_train_accuracy = result['d_train_accuracy']
                    self.train_summary_writer.add_summary(result['train_summary'], step)
                    self.train_summary_writer.flush()
                    print_msg = "[{}/{}] D_loss: {:.6f} D_accuracy: {:.6f} ".\
                                format(step, self.max_step, d_train_loss, d_train_accuracy)

                    if self.validation is True:

                        valid_data, valid_label = batch_generator.get_valid_batch()

                        fetch_dict = {
                            "d_valid_loss": self.D.loss,
                            "d_valid_accuracy": self.D.accuracy,
                            "valid_summary" : self.valid_summary_op
                        }

                        feed_dict = {
                            self.D.input_x: valid_data,
                            self.D.input_y: valid_label,
                            self.D.dropout_keep_prob: 1 # keep all units ON
                        }

                        result = sess.run(fetch_dict, feed_dict)

                        d_valid_loss = result['d_valid_loss']
                        d_valid_accuracy = result['d_valid_accuracy']
                        self.valid_summary_writer.add_summary(result['valid_summary'], step)
                        self.valid_summary_writer.flush()
                        print_msg = print_msg + "| [Valid] D_loss: {:.6f} D_accuracy: {:.6f}  ".\
                                    format(d_valid_loss, d_valid_accuracy)

                        if self.early_stopping:
                            if self.early_stopping_metric == 'loss':
                                valid_metric_history.append(d_valid_loss)
                            elif self.early_stopping_metric == 'accuracy':
                                valid_metric_history.append(d_valid_accuracy)
                            if step >= 100:
                                middle_idx = int(round(len(valid_metric_history)/2))
                                latest_idx = len(valid_metric_history) - 10
                                last_half_avg = np.mean(valid_metric_history[middle_idx:])
                                last_few_avg = np.mean(valid_metric_history[latest_idx:])
                                valid_criterion = np.absolute(last_half_avg - last_few_avg)
                                print_msg = print_msg + "valid_criterion: {:.6f}  ".\
                                            format(valid_criterion)
                                if valid_criterion <= self.early_stopping_threshold:
                                    #self.sv.request_stop()
                                    #pbar.close()
                                    #print(print_msg)
                                    #print("[!] Early stopping. End of training loop.")
                                    pass

                        print(print_msg)

                if step % (self.log_step * 10) == 0: # default : every 100 steps
                    pass

                if step % self.lr_update_step == self.lr_update_step - 1:
                    sess.run(self.d_lr_update)

    def test_interactive(self):

        with self.sv.managed_session() as sess:

            message = """
            #################################################################
            ### This is interactive test session for CharCNN Discriminator###
            #################################################################
            """
            print(message)

            from nltk.tokenize import word_tokenize
            from data import TOK_UNK
            from data import append_pads
            from data import replace_unknowns
            from data import convert_to_idx

            voca = self.word2idx.keys()
            max_len = self.max_sentence_len
            sent = ""

            while sent != 'exit':
                sent = raw_input('Test >> ')

                words = word_tokenize(sent)
                unknowns = list(set(words)-set(voca))
                unknowns_str = ', '.join(str(unknown) for unknown in unknowns)
                if len(unknowns>0):
                    print('[INFO] Unknown words : '+ unknowns_str)

                words = replace_unknowns([words], unknowns)[0]
                words_padded = append_pads([words], max_len)[0]
                input_matrix = convert_to_idx([words_padded], self.word2idx)

                fetch_dict = {
                    'score': self.D.scores,
                    'prediction': self.D.predictions
                }
                feed_dict = {
                    self.D.input_x: input_matrix,
                    self.D.dropout_keep_prob: 1
                }
                result = sess.run(fetch_dict, feed_dict)
                score = result['score'][0]
                prediction = bool(result['prediction'][0])
                #import pdb; pdb.set_trace()
                print('[result] score : {:.3f}/{:.3f}, prediction : {}'.\
                                format(score[0], score[1], prediction))
            print('End of test loop.')
                #self.word2idx
        def test(self):
            pass
