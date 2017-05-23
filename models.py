import numpy as np
import tensorflow as tf
from utils import *
slim = tf.contrib.slim

class Discriminator(object):

    def __init__(
      self, W_e_init, max_sentence_len, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, data_format, l2_reg_lambda=0.0):
        with tf.variable_scope("D") as vs:
            # Placeholders for input, output and dropout
            self.input_x = tf.placeholder(tf.int32, [None, max_sentence_len], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.W_e = tf.Variable(W_e_init, name="W_e", dtype=tf.float32)

            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)

            # Embedding layer
            with tf.name_scope("embedding"): # check if TF 1.1 support GPU operation for embedding lookup..
                self.embedded_chars = tf.nn.embedding_lookup(self.W_e, self.input_x)
                # [batch_size, max_sentence_len, embedding_size]
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, 1)
                # [batch_size, 1, max_sentence_len, embedding_size]
                # expand the channel dimension for conv2d operation
                # data_format : 'NCHW' / peforms better when using NVIDA cuDNN lib.

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters] # 300
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    # magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    #import pdb; pdb.set_trace()
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv",
                        data_format=data_format)
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b, data_format=data_format), name="relu")
                    # [batch_size, 1, max_sentence_len - filter_size + 1, 1]

                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, 1, max_sentence_len - filter_size + 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool",
                        data_format=data_format)
                    # [batch_size, num_filters*len(filter_sizes), 1, 1]
                    pooled_outputs.append(pooled)
                    # append all the feature representation from each type of filters
                    # lengths of pooled vectors are not the same due to different filter size

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 1) # concatenate along channel axis
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            # squeeze all the other dimensions.. *NOTE* check later

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores") # [batch_size, 2]
                self.predictions = tf.argmax(self.scores, 1, name="predictions")
                # No need to pass through softmax.

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss # L-2 reg is not effective here.

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.vars = tf.contrib.framework.get_variables(vs)
