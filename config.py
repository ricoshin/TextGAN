import argparse

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

arg_lists = [] # argument group will be appended in this list.
parser = argparse.ArgumentParser()

####### Argument groups ##########
        #  self, W_e_init, max_sentence_len, num_classes, vocab_size,
        #  embedding_size, filter_sizes, num_filters, data_format, l2_reg_lambda=0.0
settings_arg = add_argument_group('settings')
settings_arg.add_argument('--dataset', type=str, default='???')
settings_arg.add_argument('--is_train', type=str2bool, default=True)
settings_arg.add_argument('--use_gpu', type=str2bool, default=True)
settings_arg.add_argument('--save_step', type=int, default=5000)
settings_arg.add_argument('--log_step', type=int, default=50)

file_paths_arg = add_argument_group('file_paths')
file_paths_arg.add_argument('--load_path', type=str, default='')
file_paths_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
file_paths_arg.add_argument('--log_dir', type=str, default='logs') #
file_paths_arg.add_argument('--data_dir', type=str, default='data')

hyper_params_arg = add_argument_group('hyper_params')
hyper_params_arg.add_argument('--batch_size', type=int, default=48)
hyper_params_arg.add_argument('--z_num', type=int, default=128, choices=[64, 128])
hyper_params_arg.add_argument('--filter_', type=int, default=128, choices=[64, 128])

hyper_params_arg.add_argument('--optimizer', type=str, default='adam')
hyper_params_arg.add_argument('--d_lr', type=float, default=0.00002) # learning rate of Discriminator
hyper_params_arg.add_argument('--g_lr', type=float, default=0.00002) # learning rate of Generator
hyper_params_arg.add_argument('--lr_update_step', type=int, default=100000, choices=[100000, 75000])
hyper_params_arg.add_argument('--max_step', type=int, default=500000)

def get_config(): # when program get started, this function runs for the first.
    config, unparsed = parser.parse_known_args() # config is pre-defined object.

        setattr(config, 'data_format', data_format) # set data_format attribute in config.
    return config, unparsed # only config object is used later.
