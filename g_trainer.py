from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from data import convert_to_token
from models import Generator


class GTrainer(object):
    def __init__(self, config, train, valid, word_embd, word2idx, ans2idx):
        self.train_data = train
        self.valid_data = valid
        self.word2idx = word2idx
        self.idx2ans = {v: k for k, v in ans2idx.items()}
        self.num_examples = train[0].shape[0]
        self.max_sent_len = train[0].shape[1]
        self.config = config
        self.generator = Generator(word_embd, self.max_sent_len, len(ans2idx),
                                   True, z_dim=self.config.z_dim)

    def random_batch_generator(self, data, batch_size, num_steps):
        for i in range(num_steps):
            idx = np.random.permutation(self.num_examples)[:batch_size]
            batch_ques = data[0][idx]
            batch_ans = data[1][idx].reshape(-1)
            # z = np.random.uniform(-1, 1, [batch_size, self.config.z_dim])
            z = np.zeros([batch_size, self.config.z_dim])
            yield batch_ques, batch_ans, z

    def train(self):
        sess = tf.Session()

        opt = tf.train.AdamOptimizer(1e-4)
        train_op = opt.minimize(self.generator.pre_train_loss)

        logdir = os.path.join(self.config.log_dir, 'generator')
        sv = tf.train.Supervisor(logdir=logdir, save_model_secs=5)
        with sv.managed_session() as sess:
            for i, batch in enumerate(
                             self.random_batch_generator(self.train_data,
                                                         self.config.batch_size,
                                                         self.config.max_step)):
                if sv.should_stop():
                    break
                questions, answers, z = batch
                result = self.generator.pre_train(sess, train_op, z, answers,
                                                  questions)
                outputs = result[0]
                loss = result[1]

                if i % 100 == 0:
                    print('')
                    print('Step', i)
                    print('loss:', loss)
                    sample_size = 20
                    outputs = np.array(outputs).transpose()[:sample_size]
                    outputs = convert_to_token(outputs, self.word2idx)
                    inputs = answers[:sample_size]
                    inputs = [self.idx2ans[ans]
                              for ans in answers[:sample_size]]
                    for ans, ques in zip(inputs, outputs):
                        print('%20s => %s' % (ans, ' '.join(ques)))
