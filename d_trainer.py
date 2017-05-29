from __future__ import print_function
import os
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm
from collections import deque
from utils import *
from data import TOK_PAD
from discriminator import Discriminator
from data_loader import Dataset
from data_loader import BatchGenerator

class DTrainer(object):
    def __init__(self, config, train_data, valid_data, W_e_init, word2idx):

        self.cfg = config
        self.batch_size = config.batch_size
        self.data_train = train_data
        self.data_valid = valid_data

        self.W_e_init = np.asarray(W_e_init)
        self.word2idx = word2idx

        self.pad_idx = word2idx[TOK_PAD]
        self.max_sentence_len = np.asarray(train_data[0]).shape[1]
        self.vocab_size = self.W_e_init.shape[0]
        self.embedding_size = self.W_e_init.shape[1]
        self.filter_sizes = [3,4,5]
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.build_model()

        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter(self.cfg.model_dir+'/train')
        self.valid_writer = tf.summary.FileWriter(self.cfg.model_dir+'/valid')

        self.sv = tf.train.Supervisor(logdir=self.cfg.model_dir,
                                      is_chief=True,
                                      saver=self.saver,
                                      summary_op=None,
                                      #summary_writer=self.summary_writer,
                                      save_model_secs=self.cfg.save_model_secs,
                                      global_step=self.global_step,
                                      ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = self.sv.prepare_or_wait_for_session(config=sess_config)

    def build_model(self):

        self.D = Discriminator(W_e_init=self.W_e_init,
                               max_sentence_len=self.max_sentence_len,
                               num_classes=2,
                               vocab_size=self.vocab_size,
                               embedding_size=self.embedding_size,
                               filter_sizes=self.filter_sizes,
                               num_filters=self.cfg.d_num_filters,
                               data_format=self.cfg.data_format,
                               l2_reg_lambda=self.cfg.d_l2_reg_lambda)

        if self.cfg.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Not supported optimizer")

        self.d_lr = tf.Variable(self.cfg.d_lr, name='d_lr')
        self.d_lr_update = tf.assign(self.d_lr, (self.d_lr * 0.5),
                                     name='d_lr_update')
        d_optimizer = self.optimizer(self.cfg.d_lr)

        grads_and_vars = d_optimizer.compute_gradients(self.D.loss, self.D.vars)
        self.d_train_op = d_optimizer.apply_gradients(grads_and_vars,
                                                   global_step=self.global_step)

        # Keep track of gradient values and sparsity (optional)
        for _grads, _vars in grads_and_vars:
            if _grads is not None:
                tf.summary.histogram("{}/grad/hist".format(_vars.name), _grads,
                                     collections=['train'])
                tf.summary.scalar("{}/grad/sparsity".format(_vars.name),
                                  tf.nn.zero_fraction(_grads),
                                  collections=['train'])
        c_train = ['train']
        c_both = ['train', 'valid']
        tf.summary.scalar("D_loss", self.D.loss, collections=c_both)
        tf.summary.scalar("D_accuracy", self.D.accuracy, collections=c_both)
        tf.summary.scalar("D_learning_rate", self.d_lr, collections=c_train)

        self.train_summary_op = tf.summary.merge_all('train')
        self.valid_summary_op = tf.summary.merge_all('valid')

        self.d_lr_update = tf.assign(self.d_lr, (self.d_lr * 0.5),
                                     name='d_lr_update')

    def train(self):



        train_dataset = Dataset(self.data_train, self.batch_size,
                                self.vocab_size, self.pad_idx)
        valid_dataset = Dataset(self.data_valid, self.batch_size,
                                self.vocab_size, self.pad_idx)
        train_generator = BatchGenerator(train_dataset)
        valid_generator = BatchGenerator(valid_dataset)

        if self.cfg.early_stopping:
            valid_metric_history = []

        dropout_prob = self.cfg.d_dropout_prob
        pbar = tqdm(total = self.cfg.max_step)
        step = self.sess.run(self.global_step)

        if step > 1:
            pbar.update(step)

        while not self.sv.should_stop():

            pbar.update(1)
            que_train, ans_train = train_generator.get_d_data_batch()
            label_train = train_generator.get_d_label_batch()

            feed = [que_train, ans_train, label_train] # fetch_list
            ops = [self.d_train_op, self.global_step]
            result = self.D.run(self.sess, ops, feed, dropout_prob)
            step = result[1]

            # update learning rate (optional)
            if step % self.cfg.lr_update_step == self.cfg.lr_update_step - 1:
                self.sess.run(self.d_lr_update)

            if not step % self.cfg.log_step == 0: # default : 50
                continue

            ops = [self.train_summary_op, self.D.loss, self.D.accuracy]
            result = self.D.run(self.sess, ops, feed, dropout_prob)
            self.train_writer.add_summary(result[0], step)
            print_msg = "[{}/{}] D_loss: {:.6f} D_accuracy: {:.6f} ".\
                         format(step, self.cfg.max_step, result[1], result[2])

            if self.cfg.validation:
                que_valid, ans_valid = valid_generator.get_d_data_batch()
                label_valid = valid_generator.get_d_label_batch()

                feed = [que_valid, ans_valid, label_valid] # fetch_list
                ops = [self.valid_summary_op, self.D.loss, self.D.accuracy]
                result = self.D.run(self.sess, ops, feed, 1)
                self.valid_writer.add_summary(result[0], step)
                add_msg = "| [Valid] D_loss: {:.6f} D_accuracy: {:.6f}  ".\
                            format(result[1], result[2])
                print_msg = print_msg + add_msg

                if self.cfg.early_stopping and step>=100:
                    if self.cfg.early_stopping_metric == 'loss':
                        valid_metric_history.append(result[1])
                    elif self.cfg.early_stopping_metric == 'accuracy':
                        valid_metric_history.append(result[2])
                    if step >= 1000:
                        add_msg = self.validation_monitor(valid_metric_history)
                        print_msg = print_msg + add_msg

            print(print_msg)

            if step % (self.cfg.log_step * 10) == 0:
                pass

    def validation_monitor(self, valid_metric_history):
        middle_idx = int(round(len(valid_metric_history)/2))
        latest_idx = len(valid_metric_history) - 10
        last_half_avg = np.mean(valid_metric_history[middle_idx:])
        last_few_avg = np.mean(valid_metric_history[latest_idx:])
        valid_criterion = np.absolute(last_half_avg - last_few_avg)
        msg_str = "valid_criterion: {:.6f}  ".format(valid_criterion)
        if valid_criterion <= self.cfg.early_stopping_threshold:
            #self.sv.request_stop()
            #pbar.close()
            #print(print_msg)
            print("[!] Early stopping. End of training loop.")
        return msg_str

    def test_interactive(self):

        message = """
        ##################################################################
        ### This is interactive test session for CharCNN Discriminator ###
        ##################################################################
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
            sent = raw_input('Input >> ')

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
                self.D.dropout_prob: 1
            }
            result = self.sess.run(fetch_dict, feed_dict)
            score = result['score'][0]
            prediction = bool(result['prediction'][0])
            #import pdb; pdb.set_trace()
            print('[result] score : {:.3f}/{:.3f}, prediction : {}'.\
                            format(score[0], score[1], prediction))
        print('End of the loop.')
            #self.word2idx
        def test(self):
            pass
