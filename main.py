from __future__ import print_function

import numpy as np
import tensorflow as tf

#from gan_trainer import GANTrainer
from d_trainer import DTrainer
from g_trainer import GTrainer
from gan_trainer import GANTrainer
from generator import Generator
from config import get_config
from utils import set_logger, prepare_dirs, save_config
from data import load_skt_nugu_sample_dataset


def main(config):
    set_logger()
    prepare_dirs(config)

    # get trainer instance
    train, valid, W_e_init, word2idx = load_simple_questions_dataset(config)
    #data, W_e_init, word2idx = 0,0,0
    if config.trainer_mode == "G":
        trainer = GTrainer(config, train, valid, W_e_init, word2idx)
    elif config.trainer_mode == "D":
        trainer = DTrainer(config, train, valid, W_e_init, word2idx)
    else: # config.trainer_mode == "GAN":
        trainer = GANTrainer(config, train, valid, W_e_init, word2idx)

    if config.is_train:
        save_config(config) # save config file(params.json)
        trainer.train() # Train!
    else:
        if not config.load_path: # raise Exception when load_path unknown.
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        if config.interactive:
            trainer.test_interactive()
        else:
            trainer.test()

def main_G(config):
    train, ans2idx, word_embd, word2idx = load_skt_nugu_sample_dataset(config)
    g_trainer = GTrainer(config, train, None, word_embd, word2idx, ans2idx)
    g_trainer.train()

if __name__ == "__main__":
    config, unparsed = get_config() # get config from argument parser
    main(config)
    # main_G(config)
