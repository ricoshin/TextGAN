import numpy as np
import tensorflow as tf


class Discriminator(object):

    def __init__(self, word_embd, max_sentence_len, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, data_format,
                 l2_reg_lambda=0.0, que_fake=None, reuse=False):
        with tf.variable_scope("D", reuse=reuse) as vs:
            # Placeholders for input, output and dropout
            x_dim = [None, max_sentence_len, vocab_size]
            y_dim = [None, num_classes]
            # self.questions = tf.concat([self.que_real, que_fake],axis=0)
            if que_fake is None:
                self.questions = tf.placeholder(tf.float32, x_dim, name="que")
            else:
                self.questions = que_fake
            self.labels = tf.placeholder(tf.float32, y_dim, name="label")
            self.answers = tf.placeholder(tf.int32, [None, 1],
                                          name="ans")
            self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
            self.W_e = tf.get_variable(
                "W_e", shape=word_embd.shape,
                initializer=tf.constant_initializer(word_embd))
            self.is_pre_train = tf.placeholder(tf.bool, name="is_pre_train")
            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)

            # Embedding layer
            with tf.name_scope("embedding"):
                h = embedding_size
                V = vocab_size
                m = max_sentence_len

                questions = tf.reshape(self.questions, [-1, V])
                embed = tf.matmul(questions, self.W_e, a_is_sparse=True)
                embed = tf.reshape(embed, [-1, m, h])
                self.embed_expanded = tf.expand_dims(embed, 1)
                # [batch_size, max_sentence_len, embedding_size]
                # self.embed_expanded = tf.expand_dims(self.embed, 1)
                # [batch_size, 1, max_sentence_len, embedding_size]
                # expand the channel dimension for conv2d operation

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1,
                                    num_filters]
                    W = tf.get_variable(
                        "W", shape=filter_shape,
                        initializer=tf.truncated_normal_initializer(
                            stddev=0.1))
                    # magnitude is more than 2 standard deviations
                    #   from the mean are dropped and re-picked.
                    b = tf.get_variable(
                        "b", shape=[num_filters],
                        initializer=tf.constant_initializer(0.1))
                    conv = tf.nn.conv2d(self.embed_expanded, W,
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="conv",
                                        data_format=data_format)
                    # Apply nonlinearity
                    h = tf.nn.relu(
                            tf.nn.bias_add(conv, b, data_format=data_format),
                            name="relu")
                    # [batch_size, 1, max_sentence_len - filter_size + 1, 1]

                    # Maxpooling over the outputs
                    ksize = [1, 1, max_sentence_len - filter_size + 1, 1]
                    pooled = tf.nn.max_pool(h,
                                            ksize=ksize,
                                            strides=[1, 1, 1, 1],
                                            padding='VALID',
                                            name="pool",
                                            data_format=data_format)
                    # [batch_size, num_filters*len(filter_sizes), 1, 1]
                    pooled_outputs.append(pooled)  # 900-d
                    # append all the features from each type of filters

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 1)  # along channel axis
            self.feature = tf.reshape(self.h_pool, [-1, num_filters_total])
            # squeeze all the other dimensions.. *NOTE* check later

            # Concatenate feature vector & answer embedding
            with tf.name_scope("concat_ans"):
                ans_embed = tf.nn.embedding_lookup(self.W_e, self.answers)
                ans_embed = tf.squeeze(ans_embed)
                last_feature = tf.concat([self.feature, ans_embed], axis=1,
                                         name='feature_concat')
                last_feature_len = num_filters_total + embedding_size

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(last_feature, self.dropout_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                shape = [last_feature_len, num_classes]
                W = tf.get_variable(
                    "W", shape=shape,
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b", shape=[num_classes],
                                    initializer=tf.constant_initializer(0.1))
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                # [batch_size, 2]
                self.predictions = tf.argmax(self.scores, 1,
                                             name="predictions")
                # No need to pass through softmax.

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                                        logits=self.scores, labels=self.labels)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
                # L-2 reg is not effective here.

            # Accuracy
            with tf.name_scope("accuracy"):
                correct = tf.equal(self.predictions, tf.argmax(self.labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct, "float"),
                                               name="accuracy")

        self.vars = tf.contrib.framework.get_variables(vs)

    def run(self, sess, ops, feed_list, dropout_prob):
        """feed_list = list([quenstion, answer, label])"""
        questions, answers, labels = feed_list
        answers = np.reshape(answers, [len(answers), 1])
        feed_dict = {
            self.questions: questions,
            self.answers: answers,
            self.labels: labels,
            self.dropout_prob: dropout_prob,
            self.is_pre_train: True,
        }
        return sess.run(ops, feed_dict)
