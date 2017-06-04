from __future__ import print_function

import os
import json
import logging
from datetime import datetime


def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")


def save_config(config):
    # param_path = {model_dir}/params.json
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def set_logger():
    """called right after start of main func."""
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)


def prepare_dirs(config):
    # default log path is set to ./log/{dataset_name}_{md}_{HMS}

    if config.load_path:
        # e.g. log/my_dataset_{load_path}
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            # e.g. my_dataset_{load_path}
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:  # e.g. {load_path}
                config.model_name = "{}_{}".format(config.dataset,
                                                   config.load_path)
                # model_name = {dataset}_{my}
    # if load_path is unset (to train a model)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())
        # model_name = {dataset}_{time}

    if not hasattr(config, 'model_dir'):
        # if model_dir is unset or load_path is not {log_dir}***
        config.model_dir = os.path.join(config.log_dir, config.model_name)
        # model_dir = {log_dir}/{model_name}

    config.data_path = os.path.join(config.data_dir, config.dataset)
    # data_path = {data_dir}/{dataset}

    # make dir if not exists [log_dir / data_dir / model_dir]
    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

# When MyLogPath = logs

# if {load_path} = logs/xxx
#   {model_dir} = logs/xxx

# if {load_path} = myDataset_xxx
#   {model_dir} = logs/myDataset_xxx

# if {load_path} = xxx
#   {model_dir} = logs/MyDataSet_xxx

# if {load_path} = None
#   {model_dir} = logs/MyDataSet_time (default)

# I think it's too complicated, and much better force users to fit the form.
