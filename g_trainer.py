from __future__ import print_function

import numpy as np
import tensorflow as tf

from data import convert_to_token
from models import Generator


class GTrainer(object):
    def __init__(self, config, train, valid, word_embd, word2idx):
        self.train_data = train
        self.valid_data = valid
        self.word2idx = word2idx
        self.num_examples = train[0].shape[0]
        self.max_sent_len = train[0].shape[1]
        self.config = config
        self.generator = Generator(word_embd, self.max_sent_len, True,
                                   z_dim=self.config.z_dim)

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
        sess.run(tf.global_variables_initializer())

        for i, batch in enumerate(
                         self.random_batch_generator(self.train_data,
                                                     self.config.batch_size,
                                                     self.config.max_step)):
            questions, answers, z = batch
            result = self.generator.pre_train(sess, train_op, z, answers,
                                              questions)
            outputs = result[0]
            loss = result[1]

            if i % 100 == 0:
                print('')
                print('Step', i)
                print('loss:', loss)
                outputs = np.array(outputs).transpose()[:10]
                outputs = convert_to_token(outputs, self.word2idx)
                inputs = answers[:10]
                inputs = convert_to_token([inputs], self.word2idx)[0]
                for ans, ques in zip(inputs, outputs):
                    print('%20s => %s' % (ans, ' '.join(ques)))
