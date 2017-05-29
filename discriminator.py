import tensorflow as tf
from utils import *


class Discriminator(object):

    def __init__(self, W_e_init, max_sentence_len, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, data_format,
                 que_fake=None, l2_reg_lambda=0.0):
        with tf.variable_scope("D") as vs:
            # Placeholders for input, output and dropout
            x_dim = [None, max_sentence_len]
            y_dim = [None, num_classes]
            # self.questions = tf.concat([self.que_real, que_fake],axis=0)
            self.questions = tf.placeholder(tf.int32, x_dim, name="input_x")
            self.labels = tf.placeholder(tf.float32, y_dim, name="input_y")
            self.answers = tf.placeholder(tf.int32, [None, 1],
                                          name="condition")
            self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
            self.W_e = tf.Variable(W_e_init, name="W_e", dtype=tf.float32)
            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)

            # Embedding layer
            with tf.name_scope("embedding"):
                embed_real = tf.nn.embedding_lookup(self.W_e, self.questions)
                if que_fake is not None:
                    h = embedding_size
                    v = vocab_size
                    m = max_sentence_len

                    # que_onehot = tf.one_hot(self.questions,depth=vocab_size,axis=-1)
                    que_fake = tf.reshape(que_fake, [-1, v])
                    embed_fake = tf.matmul(que_fake, self.W_e)
                    embed_fake = tf.reshape(embed_fake, [-1, m ,h])
                    embed = tf.concat([embed_real, embed_fake], axis=0)
                else:
                    embed = embed_real

                self.embed_expanded = tf.expand_dims(embed, 1)
                # [batch_size, max_sentence_len, embedding_size]
                # self.embed_expanded = tf.expand_dims(self.embed, 1)
                # [batch_size, 1, max_sentence_len, embedding_size]
                # expand the channel dimension for conv2d operation

                #self.embed = tf.concat(embed_list, axis=0)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(
                                    filter_shape, stddev=0.1), name="W")
                    # magnitude is more than 2 standard deviations
                    #   from the mean are dropped and re-picked.
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]),
                                    name="b")
                    conv = tf.nn.conv2d(self.embed_expanded, W,
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="conv",
                                        data_format=data_format)
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b,
                                                  data_format=data_format),
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
                    pooled_outputs.append(pooled) # 900-d
                    # append all the features from each type of filters

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 1) # along channel axis
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
                shape = shape=[last_feature_len, num_classes]
                initializer = tf.contrib.layers.xavier_initializer()
                W = tf.get_variable("W", shape=shape, initializer=initializer)
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                # [batch_size, 2]
                self.predictions = tf.argmax(self.scores, 1, name="predictions")
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
        feed_dict = {
            self.questions: questions,
            self.answers: answers,
            self.labels: labels,
            self.dropout_prob: dropout_prob
        }
        return sess.run(ops, feed_dict)
