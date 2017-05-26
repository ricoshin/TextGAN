from __future__ import print_function

import numpy as np
import tensorflow as tf

from trainer import Trainer
from models import Generator
from config import get_config
from utils import *
from data import convert_to_token, load_simple_questions_dataset


def main(config):
    set_logger()
    prepare_dirs(config)

    # get trainer instance
    train, valid, W_e_init, word2idx = load_simple_questions_dataset(config)
    #data, W_e_init, word2idx = 0,0,0
    trainer = Trainer(config, train[0], W_e_init, word2idx)

    if config.is_train:
        save_config(config) # save config file(params.json)
        trainer.train() # Train!
    else:
        if not config.load_path: # raise Exception when load_path unknown.
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test() # Test!


def main_G(config):
    train, valid, word_embd, word2idx = load_simple_questions_dataset(config)
    num_examples = train[0].shape[0]
    max_sent_len = train[0].shape[1]
    batch_size = 128
    z_dim = 100
    generator = Generator(word_embd, max_sent_len, True, z_dim=z_dim)

    sess = tf.Session()
    opt = tf.train.AdamOptimizer(1e-3)
    train_op = opt.minimize(generator.pre_train_loss)
    sess.run(tf.global_variables_initializer())
    questions = train[0]
    answers = train[1]
    z = np.random.rand(batch_size, z_dim)
    num_batches = int(math.ceil(num_examples/batch_size))
    for i in range(num_batches):
        print(i)
        batch_ans = answers[i*batch_size:(i+1)*batch_size]
        batch_ques = questions[i*batch_size:(i+1)*batch_size]
        outputs, loss, _ = generator.update(sess, train_op, z, batch_ans,
                                            batch_ques)
        print('loss:', loss)
        output = np.array(outputs).transpose()[0]
        output = convert_to_token([output], word2idx)[0]
        print('-', ' '.join(output))

    outputs = np.array(outputs).transpose()[1:11]
    outputs = convert_to_token(outputs, word2idx)
    for output in outputs:
        print('-', ' '.join(output))

if __name__ == "__main__":
    config, unparsed = get_config() # get config from argument parser
    main(config)
    # main_G(config)
