from __future__ import print_function

import os
import numpy as np
from tqdm import trange
from collections import deque
from utils import *
import tensorflow as tf
from models import *

class Trainer(object):
    def __init__(self, config):
        #  self, W_e_init, max_sentence_len, num_classes, vocab_size,
        #  embedding_size, filter_sizes, num_filters, data_format, l2_reg_lambda=0.0
        self.data = FROM_HW
        self.W_e_init = FROM_HW
        self.max_sentence_len = FROM_HW
        self.vocab_size = FROM_HW
        self.embedding_size = FROM_HW

        self.filter_sizes = config.filter_sizes
        self.num_filters = config.num_filters
        self.data_format = config.data_format
        self.droput_prob = config.droput_prob

        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.z_num = config.z_num

        self.model_dir = config.model_dir # e.g. logs/MyDataSet_xxx
        self.load_path = config.load_path # e.g. logs/MyDataset_xxx(test) or None(train)

        self.use_gpu = config.use_gpu
        self.g_lr, self.d_lr = config.g_lr, config.d_lr

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train

        self.build_model()
        self.plan_optimize()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        self.sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300, # 5 minutes
                                global_step=self.myGAN.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = self.sv.prepare_or_wait_for_session(config=sess_config)


    def build_model(self):

        #  self, W_e_init, max_sentence_len, num_classes, vocab_size,
        #  embedding_size, filter_sizes, num_filters, data_format, l2_reg_lambda=0.0
        discriminator = Discriminator(self.W_e_init, self.max_sentence_len, 2,
                                        self.vocab_size, self.embedding_size, self.num_filters,
                                        self.num_filters, self.data_format, self.l2_reg_lambda)

        if optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        self.g_lr = tf.Variable(g_lr, name='g_lr')
        self.d_lr = tf.Variable(d_lr, name='d_lr')

        # What the heck?
        self.g_lr_update = tf.assign(self.g_lr, (self.g_lr * 0.5), name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, (self.d_lr * 0.5), name='d_lr_update')

        #import pdb; pdb.set_trace()
        g_optimizer, d_optimizer = self.optimizer(self.g_lr), self.optimizer(self.d_lr)


        d_optim = d_optimizer.minimize(self.TOTAL_D_loss, var_list=self.D_var)
        g_optim = g_optimizer.minimize(self.TOTAL_G_loss, global_step=self.step, var_list=self.G_var)
        # global_step: Optional Variable to increment by one after the variables have been updated.

        # Context manager : k_update will be only assigned after d_optim and g_optim have been computed.
        with tf.control_dependencies([d_optim, g_optim]):
            self.k_update = tf.assign(self.k_t, tf.clip_by_value(self.k_t + lambda_k * self.balance, 0, 1))

        self.summary_op = tf.summary.merge([

            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),

        ])

        self.g_lr_update = tf.assign(self.g_lr, (self.g_lr * 0.5), name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, (self.d_lr * 0.5), name='d_lr_update')

    def train(self):

        for step in trange(self.start_step, self.max_step): # 0 - 500000
            fetch_dict = { # evaluate every step

            }
            if step % self.log_step == 0: # default : 50
                fetch_dict.update({

                })
            result = self.sess.run(fetch_dict)

            measure = result['measure']
            measure_history.append(measure)

            if step % self.log_step == 0: # default : 50
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                g_loss = result['TOTAL_G_loss']
                d_loss = result['TOTAL_D_loss']
                k_t = result['k_t']

                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} measure: {:.4f}, k_t: {:.4f}". \
                      format(step, self.max_step, d_loss, g_loss, measure, k_t))

            if step % (self.log_step * 10) == 0: # default : every 100 steps
                pass

            if step % self.lr_update_step == self.lr_update_step - 1:
                pass

    def test(self):
        pass
