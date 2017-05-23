import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from utils import *
from data import load_simple_questions_dataset


def main(config):
    set_logger()
    prepare_dirs(config)

    # get trainer instance
    data, W_e_init, word2idx = load_simple_questions_dataset()
    #data, W_e_init, word2idx = 0,0,0
    trainer = Trainer(config, data, W_e_init, word2idx)

    if config.is_train:
        save_config(config) # save config file(params.json)
        trainer.train() # Train!
    else:
        if not config.load_path: # raise Exception when load_path unknown.
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test() # Test!

if __name__ == "__main__":
    config, unparsed = get_config() # get config from argument parser
    main(config)
