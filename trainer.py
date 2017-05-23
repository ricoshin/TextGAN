from __future__ import print_function

import os
import numpy as np
from tqdm import trange
from collections import deque
from utils import *
import tensorflow as tf
from models import *

class Trainer(object):
    def __init__(self, config, data, W_e_init):
        #  self, W_e_init, max_sentence_len, num_classes, vocab_size,
        #  embedding_size, filter_sizes, num_filters, data_format, l2_reg_lambda=0.0
        #MAX_SENTENCE_LEN = 20
        #VOCAB_SIZE = 1000
        #EMBEDDING_SIZE = 300
        self.batch_size = config.batch_size
        #self.data = np.random.randint(VOCAB_SIZE, size=(self.batch_size, MAX_SENTENCE_LEN))
        #self.W_e_init = np.random.rand(VOCAB_SIZE, EMBEDDING_SIZE)
        self.max_sentence_len = data.shape[1]
        self.vocab_size = W_e_init.shape[0]
        self.embedding_size = W_e_init.shape[1]

        def convert_to_onehot(labels):
            num_classes = np.max(labels)+1
            return np.eye(num_classes)[labels]

        self.y_real = convert_to_onehot(np.ones(self.batch_size, dtype=np.int))
        self.y_fake = convert_to_onehot(np.zeros(self.batch_size, dtype=np.int))

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
        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train



        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
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

        for step in trange(self.start_step, self.max_step): # 0 - 500000

            fetch_dict = {
                "d_train_op": self.d_train_op,
                "global_step": self.global_step,
                "D_loss": self.D.loss,
                "D_accuracy": self.D.accuracy
            }
            feed_dict = { # evaluate every step
                self.D.input_x: self.data,
                self.D.input_y: self.y_real,
                self.D.dropout_keep_prob: self.dropout_keep_prob
            }
            if step % self.log_step == 0: # default : 50
                fetch_dict.update({
                    "summary": self.summary_op
                })
            result = self.sess.run(fetch_dict, feed_dict)

            if step % self.log_step == 0: # default : 50
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                d_loss = result['D_loss']
                d_accuracy = result['D_accuracy']

                print("[{}/{}] D_loss: {:.6f} D_accuracy: {:.6f} ". \
                      format(step, self.max_step, d_loss, d_accuracy))

            if step % (self.log_step * 10) == 0: # default : every 100 steps
                pass

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run(self.d_lr_update)

    def test(self):

        pass
