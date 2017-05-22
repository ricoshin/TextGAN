import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from utils import prepare_dirs_and_logger, save_config


def main(config):
    get_logger()
    prepare_dirs(config)

    # get trainer instance
    trainer = Trainer(config)

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
