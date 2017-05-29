import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

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
settings_arg.add_argument('--dataset', type=str, default='mydataset')
settings_arg.add_argument('--is_train', type=str2bool, default=True)
settings_arg.add_argument('--trainer_mode', type=str, choices=['D','G','GAN'], default='GAN')
settings_arg.add_argument('--interactive', type=str2bool, default=True)
settings_arg.add_argument('--validation', type=str2bool, default=True)
settings_arg.add_argument('--early_stopping', type=str2bool, default=True)
settings_arg.add_argument('--early_stopping_metric', type=str, choices=['loss','accuracy'], default='accuracy')
settings_arg.add_argument('--early_stopping_threshold', type=int, default=0.001)
settings_arg.add_argument('--use_gpu', type=str2bool, default=True)
settings_arg.add_argument('--save_step', type=int, default=5000)
settings_arg.add_argument('--valid_step', type=int, default=100)
settings_arg.add_argument('--log_step', type=int, default=50)
settings_arg.add_argument('--max_step', type=int, default=1000000)
settings_arg.add_argument('--save_model_secs', type=int, default=60)
settings_arg.add_argument('--g_per_d_train', type=int, default=5)
settings_arg.add_argument('--num_samples', type=int, default=5)

file_paths_arg = add_argument_group('file_paths')
file_paths_arg.add_argument('--load_path', type=str, default='')
file_paths_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
file_paths_arg.add_argument('--log_dir', type=str, default='logs') #
file_paths_arg.add_argument('--data_dir', type=str, default='data')
file_paths_arg.add_argument('--d_path', type=str, default='')
file_paths_arg.add_argument('--g_path', type=str, default='')


hyper_params_arg = add_argument_group('hyper_params')
hyper_params_arg.add_argument('--batch_size', type=int, default=128)
hyper_params_arg.add_argument('--z_dim', type=int, default=128, choices=[64, 128])
hyper_params_arg.add_argument('--d_dropout_prob', type=float, default=0.5)
hyper_params_arg.add_argument('--d_num_filters', type=int, default=300)
hyper_params_arg.add_argument('--d_l2_reg_lambda', type=float, default=0)

hyper_params_arg.add_argument('--optimizer', type=str, default='adam')
hyper_params_arg.add_argument('--d_lr', type=float, default=0.00002) # learning rate of Discriminator
hyper_params_arg.add_argument('--g_lr', type=float, default=0.00002) # learning rate of Generator
hyper_params_arg.add_argument('--lr_update_step', type=int, default=100000, choices=[100000, 75000])

def get_config(): # when program get started, this function runs for the first.
    config, unparsed = parser.parse_known_args() # config is pre-defined object.
    if config.use_gpu: # NCHW on GPU / NHWC on CPU

        data_format = 'NCHW' # cuDNN default image data format
                             # [batch_size, channels, height, width]
                             # although cuDNN can operate on both formats,
                             #  it's faster to operate in its default format.
    else:
        data_format = 'NHWC' # TensorFlow default image data format
                             # [batch_size, height, width, channels]
                             # NHWC is a little faster on CPU

    setattr(config, 'data_format', data_format) # set data_format attribute in config.
    return config, unparsed # only config object is used later.
