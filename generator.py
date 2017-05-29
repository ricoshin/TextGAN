import numpy as np
import tensorflow as tf


class Generator(object):
    def __init__(self, word_embd, max_ques_len, num_answers, is_pre_train,
                 z_dim=100, hid_dim=100):
        self.is_pre_train = is_pre_train
        with tf.variable_scope('G'):
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            word_embd = tf.Variable(word_embd, name='word_embd',
                                    dtype=tf.float32)
            vocab_size, word_embd_size = word_embd.get_shape().as_list()
            self.z = tf.placeholder(tf.float32,
                                    shape=[None, z_dim],
                                    name='z')
            self.answers = tf.placeholder(tf.int32,
                                          shape=[None],
                                          name='answers')
            self.targets = tf.placeholder(tf.int32,
                                          shape=[None, max_ques_len],
                                          name='targets')
            V = tf.Variable(tf.random_normal([hid_dim, vocab_size]), name='V')

            # C = tf.Variable(tf.random_normal([z_dim+word_embd_size, hid_dim]),
            #                 name='C')
            C = tf.Variable(tf.random_normal([z_dim+num_answers, hid_dim]),
                            name='C')

            # # [batch_size, word_embd_size]
            # ans_embd = tf.nn.embedding_lookup(word_embd, self.answers,
            #                                   name='ans_embd')
            # # [batch_size, z_dim + word_embd_size]
            # za = tf.concat([self.z, ans_embd], axis=1, name='za')

            # [batch_size, num_answers]
            ans_onehot = tf.one_hot(self.answers, num_answers, axis=-1)
            # [batch_size, z_dim + num_answers]
            za = tf.concat([self.z, ans_onehot], axis=1, name='za')

            # [batch_size, hid_dim]
            h_1 = tf.matmul(za, C)
            # [batch_size, vocab_size]
            hV_1 = tf.matmul(h_1, V)
            # [batch_size]

            # w_1 = tf.argmax(hV_1, axis=1)
            w_1 = tf.nn.softmax(hV_1*10000)
            self.outputs = [w_1]

            # [batch_size, word_embd_size]
            # y_1 = tf.nn.embedding_lookup(word_embd, w_1)
            y_1 = tf.matmul(w_1, word_embd)

            cell = tf.contrib.rnn.LSTMCell(hid_dim)

            state = cell.zero_state(self.batch_size, dtype=tf.float32)
            state = tf.contrib.rnn.LSTMStateTuple(state.c, h_1)

            pre_train_losses = []
            if self.is_pre_train:
                inputs = tf.nn.embedding_lookup(word_embd, self.targets[:, 0])
                # inputs = tf.stop_gradient(inputs)

                labels = tf.one_hot(self.targets[:, 0], vocab_size, axis=-1)
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=hV_1,
                                                               labels=labels)
                pre_train_losses.append(loss)
            else:
                inputs = y_1

            for t in range(1, max_ques_len):
                output, state = cell(inputs, state)
                hV = tf.matmul(output, V)

                # w_t = tf.argmax(hV, axis=1)
                w_t = tf.nn.softmax(hV*10000)
                self.outputs.append(w_t)
                # y_t = tf.nn.embedding_lookup(word_embd, w_t)
                y_t = tf.matmul(w_t, word_embd)

                if self.is_pre_train:
                    inputs = tf.nn.embedding_lookup(word_embd,
                                                    self.targets[:, t])
                    labels = tf.one_hot(self.targets[:, t], vocab_size, axis=-1)
                    loss = tf.nn.softmax_cross_entropy_with_logits(
                        logits=hV, labels=labels)
                    pre_train_losses.append(loss)
                else:
                    inputs = y_t
                tf.get_variable_scope().reuse_variables()
            self.pre_train_loss = tf.reduce_mean(pre_train_losses,
                                                 name='pre_train_loss')

    def run(self, sess, train_op, z, answers, targets=None):
        if self.is_pre_train and targets is None:
            raise Exception('targets needed when pre-training')
        batch_size = answers.shape[0]
        feed_dict = {
            self.batch_size: batch_size,
            self.z: z,
            self.answers: np.reshape(answers, [batch_size]),
            self.targets: targets,
        }
        if self.is_pre_train:
            return sess.run([self.outputs, self.pre_train_loss, train_op],
                            feed_dict)
        else:
            outputs = sess.run(self.outputs, feed_dict)
        return outputs
