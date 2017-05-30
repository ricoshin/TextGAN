from __future__ import print_function

import numpy as np
#from tqdm import trange
from tqdm import tqdm
from collections import deque
from utils import *
import tensorflow as tf
from discriminator import Discriminator
from generator import Generator
from data import TOK_PAD, convert_to_token
import random


class GANTrainer(object):
    def __init__(self, config, train_data, valid_data, W_e_init, word2idx,
                 ans2idx):

        self.cfg = config
        self.data_train = train_data
        self.data_valid = valid_data

        self.W_e_init = np.asarray(W_e_init)
        self.word2idx = word2idx
        self.ans2idx = ans2idx
        if ans2idx != None:
            self.idx2ans = {v: k for k, v in ans2idx.items()}
        self.pad_idx = word2idx[TOK_PAD]
        self.max_sentence_len = np.asarray(train_data[0]).shape[1]
        self.vocab_size = self.W_e_init.shape[0]
        self.embedding_size = self.W_e_init.shape[1]
        self.filter_sizes = [3,4,5]
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.build_model()

        self.saver = tf.train.Saver()

        pretrain_g_saver = tf.train.Saver(self.G.vars)
        pretrain_d_saver = tf.train.Saver(self.D_fake.vars)

        def load_pretrain(sess):
            #import pdb; pdb.set_trace()
            pretrain_g_saver.restore(sess, self.cfg.g_path)
            print("[!] Pre-trained generator chkpt loaded : ", self.cfg.g_path)
            pretrain_d_saver.restore(sess, self.cfg.d_path)
            print("[!] Pre-trained discriminator chkpt loaded : ", self.cfg.d_path)

        self.writer = tf.summary.FileWriter(self.cfg.model_dir)

        self.sv = tf.train.Supervisor(logdir=self.cfg.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                init_fn=load_pretrain,
                                #summary_writer=self.summary_writer,
                                save_model_secs=self.cfg.save_model_secs,
                                global_step=self.global_step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = self.sv.prepare_or_wait_for_session(config=sess_config)


    def compute_generator_loss(self, feature_real, feature_fake):
        pass

    def compute_generator_loss_tmp(self, feature_real, feature_fake):

        #feature_real = d_feature[0:self.cfg.batch_size]
        #feature_fake = d_feature[self.cfg.batch_size:]
        return tf.reduce_mean(tf.square(tf.subtract(feature_real,feature_fake)))

    def build_model(self):
        #  self, W_e_init, max_sentence_len, num_classes, vocab_size,
        #  embedding_size, filter_sizes, num_filters, data_format, l2_reg_lambda=0.0
        is_onehot = True if self.cfg.dataset == 'nugu' else False
        self.G = Generator(word_embd=self.W_e_init,
                           ans2idx=self.ans2idx,
                           max_ques_len=self.max_sentence_len,
                           is_pre_train=False,
                           is_onehot=is_onehot,
                           z_dim=self.cfg.z_dim)

        self.D_fake = Discriminator(W_e_init=self.W_e_init,
                                    max_sentence_len=self.max_sentence_len,
                                    num_classes=2,
                                    vocab_size=self.vocab_size,
                                    embedding_size=self.embedding_size,
                                    filter_sizes=self.filter_sizes,
                                    num_filters=self.cfg.d_num_filters,
                                    data_format=self.cfg.data_format,
                                    l2_reg_lambda=self.cfg.d_l2_reg_lambda,
                                    que_fake=self.G.outputs)

        self.D_real = Discriminator(W_e_init=self.W_e_init,
                                    max_sentence_len=self.max_sentence_len,
                                    num_classes=2,
                                    vocab_size=self.vocab_size,
                                    embedding_size=self.embedding_size,
                                    filter_sizes=self.filter_sizes,
                                    num_filters=self.cfg.d_num_filters,
                                    data_format=self.cfg.data_format,
                                    l2_reg_lambda=self.cfg.d_l2_reg_lambda,
                                    que_fake=None,
                                    reuse=True)


        self.g_loss = self.compute_generator_loss_tmp(self.D_real.feature,
                                                      self.D_fake.feature)#D.feature)
        self.d_loss = (self.D_real.loss + self.D_fake.loss)/2
        self.d_accuracy = (self.D_real.accuracy + self.D_fake.accuracy)/2

        if self.cfg.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Not supported optimizer")

        self.g_lr = tf.Variable(self.cfg.g_lr, name='g_lr')
        self.d_lr = tf.Variable(self.cfg.d_lr, name='d_lr')

        g_optimizer = tf.train.AdamOptimizer(self.g_lr)
        d_optimizer = self.optimizer(self.d_lr)
        #import pdb; pdb.set_trace()
        d_grads_vars = d_optimizer.compute_gradients(self.d_loss,
                                                     self.D_real.vars)
        g_grads_vars = g_optimizer.compute_gradients(self.g_loss, self.G.vars)
        self.d_train_op = d_optimizer.apply_gradients(d_grads_vars)
        self.g_train_op = g_optimizer.apply_gradients(g_grads_vars,
                                                 global_step=self.global_step)

        # Keep track of gradient values and sparsity (optional)
        grads_vars = g_grads_vars + d_grads_vars
        for _grads, _vars in grads_vars:
            if _grads is not None:
                tf.summary.histogram("{}/grad/hist".format(_vars.name), _grads)
                tf.summary.scalar("{}/grad/sparsity".format(_vars.name),
                                  tf.nn.zero_fraction(_grads))

        tf.summary.scalar("Loss/G_loss", self.g_loss)
        tf.summary.scalar("Loss/D_loss", self.d_loss)
        tf.summary.scalar("misc/D_accuracy", self.d_accuracy)
        tf.summary.scalar("misc/G_learning_rate", self.g_lr)
        tf.summary.scalar("misc/D_learning_rate", self.d_lr)

        self.summary_op = tf.summary.merge_all()
        self.g_lr_update = tf.assign(self.g_lr, (self.g_lr * 0.5),
                                     name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, (self.d_lr * 0.5),
                                     name='d_lr_update')

    def run_gan(self, sess, ops, feed_list):
        """feed_list = list([quenstions, answers, labels, z])"""
        questions, answers, z, dropout_prob = feed_list
        batch_size = answers.shape[0]
        answers_g = np.reshape(answers, [batch_size])   # *NOTE*answer shape
        answers_d = np.reshape(answers, [batch_size, 1])# better be unified!
        labels_real = self.label_real
        labels_fake = self.label_fake
        feed_dict = {
            self.G.batch_size: batch_size,
            self.G.z: z,
            self.G.answers: answers_g,
            self.G.targets: np.argmax(questions, axis=-1),
            self.D_real.questions: questions,
            self.D_real.answers: answers_d,
            self.D_real.labels: labels_real,
            self.D_real.dropout_prob: dropout_prob,
            self.D_fake.answers: answers_d,
            self.D_fake.labels: labels_fake,
            self.D_fake.dropout_prob: dropout_prob,
        }
        return sess.run(ops, feed_dict)


    def train(self):

        from data_loader import Dataset
        from data_loader import BatchGenerator

        train_dataset = Dataset(self.data_train, self.cfg.batch_size,
                                self.vocab_size, self.pad_idx)
        train_generator = BatchGenerator(train_dataset)
        self.label_real = train_generator.get_binary_label_batch(True)
        self.label_fake = train_generator.get_binary_label_batch(False)

        dropout_prob = self.cfg.d_dropout_prob
        pbar = tqdm(total = self.cfg.max_step)
        step = self.sess.run(self.global_step)

        z_test = np.random.uniform(-1, 1,[self.cfg.batch_size, self.cfg.z_dim])

        if step > 1:
            pbar.update(step)

        while not self.sv.should_stop():
            pbar.update(1)
            que_real, ans_real = train_generator.get_gan_data_batch()

            # G train
            z = np.random.uniform(-1, 1, [self.cfg.batch_size, self.cfg.z_dim])
            feed = [que_real, ans_real, z, 1]
            ops = [self.global_step, self.g_loss, self.g_train_op]
            step, g_loss, _ = self.run_gan(self.sess, ops, feed)

            # D train
            if not step % self.cfg.g_per_d_train == 0:
                continue
            z = np.random.uniform(-1, 1, [self.cfg.batch_size, self.cfg.z_dim])
            feed = [que_real, ans_real, z, dropout_prob]
            ops = [self.d_loss,self.summary_op, self.d_train_op]
            d_loss, summary, _ = self.run_gan(self.sess, ops, feed)

            # summary & print message
            if not step % (self.cfg.g_per_d_train*10) == 0:
                continue
            print_msg = "[{}/{}] G_loss: {:.6f} D_loss: {:.6f} ".\
                         format(step, self.cfg.max_step, g_loss, d_loss)
            print(print_msg)
            self.writer.add_summary(summary, step)

            # print generated samples
            feed = [que_real, ans_real, z_test, 1]
            if self.cfg.dataset == 'nugu':
                self._print_nugu_samples(feed)
            elif self.cfg.dataset == 'simque':
                self._print_simque_samples(feed)
            else:
                raise Exception('Unsupported dataset:', self.cfg.dataset)


    def _print_simque_samples(self, z_test):
        print('Simque sample')


    def _print_nugu_samples(self, feed):
        ans_real = feed[1]
        outputs = self.run_gan(self.sess, self.G.outputs, feed)
        answers = [self.idx2ans[ans]
                   for ans in ans_real[:self.cfg.num_samples]]
        outputs = np.argmax(outputs[:self.cfg.num_samples], axis=-1)
        outputs = convert_to_token(outputs, self.word2idx)

        for ans, ques in zip(answers, outputs):
            print('%20s => %s' % (ans, ' '.join(ques)))
