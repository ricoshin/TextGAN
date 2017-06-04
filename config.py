import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser()

settings_arg = parser.add_argument_group('settings')
settings_arg.add_argument('--dataset', type=str, choices=['nugu', 'simque'])
settings_arg.add_argument('--is_train', type=str2bool, default=True)
settings_arg.add_argument('--trainer_mode', type=str,
                          choices=['D', 'G', 'GAN'], default='GAN')
settings_arg.add_argument('--interactive', type=str2bool, default=True)
settings_arg.add_argument('--validation', type=str2bool, default=True)
settings_arg.add_argument('--early_stopping', type=str2bool, default=True)
settings_arg.add_argument('--early_stopping_metric', type=str,
                          choices=['loss', 'accuracy'], default='accuracy')
settings_arg.add_argument('--early_stopping_threshold',
                          type=int, default=0.001)
settings_arg.add_argument('--use_gpu', type=str2bool, default=True)
settings_arg.add_argument('--save_step', type=int, default=5000)
settings_arg.add_argument('--valid_step', type=int, default=100)
settings_arg.add_argument('--log_step', type=int, default=25)
settings_arg.add_argument('--max_step', type=int, default=1000000)
settings_arg.add_argument('--save_model_secs', type=int, default=60)
settings_arg.add_argument('--g_per_d_train', type=int, default=5)
settings_arg.add_argument('--num_samples', type=int, default=20)

file_paths_arg = parser.add_argument_group('file_paths')
file_paths_arg.add_argument('--load_path', type=str, default='')
file_paths_arg.add_argument('--log_level', type=str, default='INFO',
                            choices=['INFO', 'DEBUG', 'WARN'])
file_paths_arg.add_argument('--log_dir', type=str, default='logs')
file_paths_arg.add_argument('--data_dir', type=str, default='data')
file_paths_arg.add_argument('--d_path', type=str,
                            default='./logs/discriminator')
file_paths_arg.add_argument('--g_path', type=str, default='./logs/generator')


hyper_params_arg = parser.add_argument_group('hyper_params')
hyper_params_arg.add_argument('--batch_size', type=int, default=128)
hyper_params_arg.add_argument('--z_dim', type=int, default=128,
                              choices=[64, 128])
hyper_params_arg.add_argument('--d_dropout_prob', type=float, default=0.5)
hyper_params_arg.add_argument('--d_num_filters', type=int, default=300)
hyper_params_arg.add_argument('--d_l2_reg_lambda', type=float, default=0)

hyper_params_arg.add_argument('--optimizer', type=str, default='adam')
hyper_params_arg.add_argument('--d_lr', type=float, default=0.00002)
hyper_params_arg.add_argument('--g_lr', type=float, default=0.00005)
hyper_params_arg.add_argument('--lr_update_step', type=int, default=100000,
                              choices=[100000, 75000])


def get_config():
    """when program get started, this function runs for the first."""
    # config is pre-defined object.
    config, unparsed = parser.parse_known_args()
    if config.use_gpu:  # NCHW on GPU / NHWC on CPU
        """cuDNN default image data format
        [batch_size, channels, height, width]
        although cuDNN can operate on both formats,
        it's faster to operate in its default format."""
        data_format = 'NCHW'
    else:
        """TensorFlow default image data format
        [batch_size, height, width, channels]
        NHWC is a little faster on CPU"""
        data_format = 'NHWC'
    # set data_format attribute in config.
    setattr(config, 'data_format', data_format)
    # only config object is used later.
    return config, unparsed
