from __future__ import print_function

import numpy as np
import tensorflow as tf

from data import convert_to_token
from generator import Generator


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
        if self.config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("Not supported optimizer:", self.config.optimizer)

        opt = tf.train.AdamOptimizer(self.config.g_lr)
        train_op = opt.minimize(self.generator.pre_train_loss)

        sv = tf.train.Supervisor(logdir=self.config.model_dir,
                                 save_model_secs=self.config.save_model_secs)
        with sv.managed_session() as sess:
            for i, batch in enumerate(
                             self.random_batch_generator(self.train_data,
                                                         self.config.batch_size,
                                                         self.config.max_step)):
                if sv.should_stop():
                    break
                questions, answers, z = batch
                outputs, loss, _ = self.generator.run(sess, train_op, z,
                                                      answers, questions)

                if i % self.config.log_step == 0:
                    print('')
                    print('Step', i)
                    print('loss:', loss)
                    outputs = np.argmax(outputs[:self.config.num_samples],
                                        axis=-1)
                    outputs = convert_to_token(outputs, self.word2idx)
                    inputs = answers[:self.config.num_samples]
                    inputs = [self.idx2ans[ans]
                              for ans in answers[:self.config.num_samples]]
                    for ans, ques in zip(inputs, outputs):
                        print('%20s => %s' % (ans, ' '.join(ques)))
