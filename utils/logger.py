import os
import sys
import copy
import yaml
import logging
import json
from datetime import datetime
from json import JSONDecodeError

from common.registry import Registry


def modify_config_file(path, config):
    """
    load .cfg file at path and modify it according to the config parameters
    """
    assert(os.path.exists(path)), AssertionError(f"Simulator configuration at {path} not exists")
    param = config['world']
    logger_param = config['logger']

    if config['command']['world'] == 'cityflow':
        with open(path, 'r') as f:
            path_config = json.load(f)
        for k in path_config.keys():
            # modify config step1
            if param.get(k) is not None:
                path_config[k] = param.get(k)
        # modify config step2
        file_name = os.path.join(get_output_file_path(config),  logger_param['replay_dir'])
        if config['world']['dir'] in file_name:
            file_name = file_name.strip(f"{config['world']['dir']} + '\n'")
        path_config['roadnetLogFile'] = file_name + f"/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.json"
        path_config['replayLogFile'] = file_name + f"/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.txt"
        with open(path, 'w') as f:
            json.dump(path_config, f, indent=2)
        
    elif config['command']['world'] == 'sumo':
        with open(path, 'r') as f:
            path_config = json.load(f)
        # config step 1
        for k in path_config.keys():
            if param.get(k) is not None:
                path_config[k] = param.get(k)
        # config step 2
        #path_config['roadnetLogFile'] = file_name + f"/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.json"
        #path_config['replayLogFile'] = file_name + f"/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.txt"
        path_config['interval'] = param['interval']
        with open(path, 'w') as f:
            json.dump(path_config, f, indent=2)


    elif config['command']['world'] == 'openengine':
        # not in .json format
        with open(path, 'r') as f:
            contents = f.readlines()
        for idx, l in enumerate(contents):
            if '=' in l:
                lhs, _ = l.split('=')
                # TODO: check interval==10 here
                if lhs.strip() in param.keys() and lhs.strip() != 'interval':
                    rhs = ' ' + str(param[lhs.strip()]) + '\n'
                    contents[idx] = lhs + '=' + rhs
                # config step 2
                if lhs.strip() == 'max_time_epoch':
                    rhs = ' ' + str(config['trainer']['steps']) + '\n'
                    contents[idx] = lhs + '=' + rhs
            elif ':' in l:
                lhs, _ = l.split(':')
                if lhs.strip() == 'report_log_mode':
                    rhs = ' ' + str(param[lhs.strip()]) + '\n'
                    contents[idx] = lhs + ':' + rhs
                if lhs.strip() == 'report_log_addr':
                    file_name = get_output_file_path(config) + '/' +  logger_param['replay_dir'] 
                    path_config['roadnetLogFile'] = file_name + f"/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.json"
                    rhs = ' ' + 'data/output_data/' + config['command']['task'] + '/'\
                        + f"{config['command']['world']}_{config['command']['agent']}_{config['command']['prefix']}"\
                            + '/' +  logger_param['replay_dir'] + '\n'
                    contents[idx] = lhs + ':' + rhs
        with open(path, 'w') as f:
            f.writelines(contents)
    else:
        raise NotImplementedError('Simulator environment not implemented')
    
    # config other world settings
    other_world_settings = dict()
    for k in param.keys():
        if k not in path_config.keys():
            other_world_settings[k] = param.get(k)
    return other_world_settings

def build_config(args):
    """
    process command line arguments and parameters stored in .yaml files.
    position args:
    -args: command line arguments take in from run.py
    """
    config_snapshot = getattr(args, 'config_snapshot', None)
    if config_snapshot is not None:
        with open(config_snapshot, 'r', encoding='utf-8') as snapshot_file:
            saved = json.load(snapshot_file)
        required_sections = ('model', 'trainer', 'logger', 'world')
        missing = [section for section in required_sections if section not in saved]
        if missing:
            raise ValueError(
                f"Config snapshot {config_snapshot} is missing sections: {missing}"
            )
        config = {
            section: copy.deepcopy(saved[section])
            for section in required_sections
        }
        duplicates_warning = {}
    else:
        agent_name = os.path.join('./configs', args.task, f'{args.agent}.yml')
        config, duplicates_warning = load_config(agent_name)
    config.update({'command': args.__dict__})
    if args.seed is not None:
        config['world']['seed'] = args.seed
    reward_mode = getattr(args, 'reward_mode', None)
    reward_mode_aliases = {
        'mean_waiting': 'queue',
        'waiting': 'queue',
        'mplight': 'queue',
        'pressure': 'pressure_abs',
    }
    if reward_mode is not None:
        reward_mode = reward_mode_aliases.get(str(reward_mode).lower(), reward_mode)
    model_overrides = {
        'agent_embedding_mode': getattr(args, 'agent_embedding_mode', None),
        'hyper_actor_arch': getattr(args, 'hyper_actor_arch', None),
        'actor_hidden1': getattr(args, 'hyper_actor_hidden1', None),
        'actor_hidden2': getattr(args, 'hyper_actor_hidden2', None),
        'hyper_adapter_mode': getattr(args, 'hyper_adapter_mode', None),
        'hyper_critic_adapter_mode': getattr(args, 'hyper_critic_adapter_mode', None),
        'hyper_film_scale': getattr(args, 'hyper_film_scale', None),
        'reward_mode': reward_mode,
        'pressure_balance_coef': getattr(args, 'pressure_balance_coef', None),
        'native_use_agent_id': getattr(args, 'native_use_agent_id', None),
        'native_agent_id_mode': getattr(args, 'native_agent_id_mode', None),
        'native_actor_arch': getattr(args, 'native_actor_arch', None),
        'native_value_arch': getattr(args, 'native_value_arch', None),
        'iru_steps': getattr(args, 'iru_steps', None),
        'iru_actor_steps': getattr(args, 'iru_actor_steps', None),
        'iru_value_steps': getattr(args, 'iru_value_steps', None),
        'iru_hidden_dim': getattr(args, 'iru_hidden_dim', None),
        'iru_num_blocks': getattr(args, 'iru_num_blocks', None),
        'profile_performance': getattr(args, 'profile_performance', None),
        'hyper_residual': getattr(args, 'hyper_residual', None),
        'hyper_residual_mode': getattr(args, 'hyper_residual_mode', None),
        'hyper_residual_scale': getattr(args, 'hyper_residual_scale', None),
        'hyper_residual_actor_scale': getattr(args, 'hyper_residual_actor_scale', None),
        'hyper_residual_value_scale': getattr(args, 'hyper_residual_value_scale', None),
        'hyper_lora_rank': getattr(args, 'hyper_lora_rank', None),
        'hyper_lora_actor_rank': getattr(args, 'hyper_lora_actor_rank', None),
        'hyper_lora_value_rank': getattr(args, 'hyper_lora_value_rank', None),
        'hyper_lora_bias': getattr(args, 'hyper_lora_bias', None),
        'hyper_head_mode': getattr(args, 'hyper_head_mode', None),
        'hyper_chunk_size': getattr(args, 'hyper_chunk_size', None),
        'hyper_chunk_embed_dim': getattr(args, 'hyper_chunk_embed_dim', None),
    }
    for key, value in model_overrides.items():
        if value is not None:
            config['model'][key] = value
    trainer_overrides = {
        'episodes': getattr(args, 'episodes', None),
        'resume_episode': getattr(args, 'resume_episode', None),
    }
    for key, value in trainer_overrides.items():
        if value is not None:
            config['trainer'][key] = value
    save_rate = getattr(args, 'save_rate', None)
    if save_rate is not None:
        config['logger']['save_rate'] = save_rate
    iru_steps = getattr(args, 'iru_steps', None)
    if iru_steps is not None:
        if getattr(args, 'iru_actor_steps', None) is None:
            config['model']['iru_actor_steps'] = iru_steps
        if getattr(args, 'iru_value_steps', None) is None:
            config['model']['iru_value_steps'] = iru_steps
    hypernet_type = getattr(args, 'hypernet_type', None)
    if hypernet_type is not None:
        config['model']['hypernet_type'] = hypernet_type
        config['model']['actor_hypernet_type'] = hypernet_type
        config['model']['critic_hypernet_type'] = hypernet_type
        config['model']['value_hypernet_type'] = hypernet_type
    return config, duplicates_warning

def load_config(path, previous_includes=[]):
    """
    process individual .yaml file and eliminate duplicate parameters
    position args:
    -path: path of .yml file
    -previous_includes: list of .yml already processed
    """
    if path in previous_includes:
        raise ValueError(
            f"Cyclic configs include detected. {path} included in previous {previous_includes}"
        )
    previous_includes = previous_includes + [path]
    direct_config = yaml.load(open(path, "r"), Loader=yaml.Loader)
    # Load configs from included files.
    if "includes" in direct_config:
        includes = direct_config.pop("includes")
    else:
        includes = []
    if not isinstance(includes, list):
        raise AttributeError(
            "Includes must be a list, '{}' provided".format(type(includes))
        )
    config = {}
    duplicates_warning = {}
    # process config recursively
    for include in includes:
        include_config, inc_dup_warning = load_config(
            include, previous_includes
        )
        duplicates_warning.update(inc_dup_warning)
        config, duplicates = merge_dicts(config, include_config)
        duplicates_warning.update(duplicates)
    config, merge_dup_warning = merge_dicts(config, direct_config)
    duplicates_warning.update(merge_dup_warning)
    return config, duplicates_warning

def merge_dicts(dict1, dict2):
    """
    merge dict2 into dict1, and dict1 will not be overwrite by dict2
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = {}

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                if k not in duplicates.keys():
                    duplicates.update({k: duplicates_k})
            else:
                return_dict[k] = dict2[k]
                duplicates.update({k: v})
    return return_dict, duplicates

def load_config_dict(config_path, other_world_settings=None):
    """
    load .cfg file at config_path
    """
    try:
        with open(config_path, 'r') as f:
            path_config = json.load(f)
    except JSONDecodeError:
        with open(config_path, 'r') as f:
            contents = f.readlines()
            path_config = {}
            for l in contents:
                if ':' in l:
                    lhs, rhs = l.split(':')
                    try:
                        val = eval(rhs.strip().strip('\n'))
                    except NameError:
                        val = rhs.strip().strip('\n')
                    path_config.update({lhs.strip().strip('\n'): val})
                if '=' in l:
                    lhs, rhs = l.split('=')
                    try:
                        val = eval(rhs.strip().strip('\n'))
                    except NameError:
                        val = rhs.strip().strip('\n')
                    path_config.update({lhs.strip().strip('\n'): val})
    if other_world_settings is not None:
        path_config.update(other_world_settings)
    return path_config

def get_output_file_path(config):
    """"
    set output path
    """
    param = config['command']
    path = os.path.join(config['world']['dir'] , 'output_data', param['task'], 
        f"{param['world']}_{param['agent']}", param['network'], param['prefix'])
    return path


class SeverityLevelBetween(logging.Filter):
    def __init__(self, min_level, max_level):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record):
        return self.min_level <= record.levelno < self.max_level

def setup_logging(level):
    root = logging.getLogger()

    # Perform setup only if logging has not been configured
    if not root.hasHandlers():
        root.setLevel(level)
        log_formatter = logging.Formatter(
            "%(asctime)s (%(levelname)s): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Send INFO to stdout
        handler_out = logging.StreamHandler(sys.stdout)
        handler_out.addFilter(
            SeverityLevelBetween(logging.INFO, logging.WARNING)
        )
        handler_out.setFormatter(log_formatter)
        root.addHandler(handler_out)

        # Send WARNING (and higher) to stderr
        handler_err = logging.StreamHandler(sys.stderr)
        handler_err.setLevel(logging.WARNING)
        handler_err.setFormatter(log_formatter)
        root.addHandler(handler_err)

        logger_dir = os.path.join(
            Registry.mapping['logger_mapping']['path'].path,
            Registry.mapping['logger_mapping']['setting'].param['log_dir'])
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)

        handler_file = logging.FileHandler(os.path.join(
            logger_dir,
            f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}_BRF.log"), mode='w'
        )
        handler_file.setLevel(level)  # TODO: SET LEVEL
        root.addHandler(handler_file)
    return root
