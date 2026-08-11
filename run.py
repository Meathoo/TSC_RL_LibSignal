import task
import trainer
import agent
import dataset
from common.registry import Registry
from common import interface
from common.utils import *
from utils.logger import *
import time
from datetime import datetime
import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# parseargs
parser = argparse.ArgumentParser(description='Run Experiment')
parser.add_argument('--thread_num', type=int, default=4, help='number of threads')  # used in cityflow
parser.add_argument('--ngpu', type=str, default="-1", help='gpu to be used')  # choose gpu card
parser.add_argument('--prefix', type=str, default='test', help="the number of prefix in this running process")
parser.add_argument('--seed', type=int, default=None, help="seed for pytorch backend")
parser.add_argument('--debug', nargs='?', const=True, default=False, type=str2bool)
parser.add_argument('--no-debug', action='store_false', dest='debug')
parser.add_argument('--interface', type=str, default="libsumo", choices=['libsumo','traci'], help="interface type") # libsumo(fast) or traci(slow)
parser.add_argument('--delay_type', type=str, default="apx", choices=['apx','real'], help="method of calculating delay") # apx(approximate) or real

parser.add_argument('-t', '--task', type=str, default="tsc", help="task type to run")
parser.add_argument('-a', '--agent', type=str, default="dqn", help="agent type of agents in RL environment")
parser.add_argument('-w', '--world', type=str, default="cityflow", choices=['cityflow','sumo'], help="simulator type")
parser.add_argument('-n', '--network', type=str, default="cityflow1x1", help="network name")
parser.add_argument('-d', '--dataset', type=str, default='onfly', help='type of dataset in training process')
parser.add_argument(
    '--agent_embedding_mode',
    type=str,
    default=None,
    choices=['one_hot', 'learned', 'topology', 'learned_topology', 'one_hot_topology'],
    help='override model.agent_embedding_mode',
)
parser.add_argument('--hypernet_type', type=str, default=None, choices=['mlp', 'linear'],
                    help='override model hypernetwork type')
parser.add_argument('--hyper_actor_arch', type=str, default=None, choices=['mlp', 'iru'],
                    help='override model.hyper_actor_arch')
parser.add_argument('--hyper_actor_hidden1', type=int, default=None,
                    help='override model.actor_hidden1 for HyperLight MLP controls')
parser.add_argument('--hyper_actor_hidden2', type=int, default=None,
                    help='override model.actor_hidden2 for HyperLight MLP controls')
parser.add_argument('--hyper_adapter_mode', type=str, default=None,
                    choices=['generated', 'film', 'none'],
                    help='override actor adaptation: generated weights, FiLM, or shared actor')
parser.add_argument('--hyper_critic_adapter_mode', type=str, default=None,
                    choices=['generated', 'film'],
                    help='override critic adaptation: generated weights or shared critic with FiLM')
parser.add_argument('--hyper_film_scale', type=float, default=None,
                    help='override model.hyper_film_scale')
parser.add_argument('--reward_mode', type=str, default=None,
                    choices=['queue', 'pressure_abs', 'queue_pressure', 'pressure', 'waiting', 'mean_waiting', 'mplight'],
                    help='override model.reward_mode')
parser.add_argument('--pressure_balance_coef', type=float, default=None,
                    help='override model.pressure_balance_coef')
parser.add_argument('--native_use_agent_id', type=str2bool, nargs='?', const=True, default=None,
                    help='override model.native_use_agent_id')
parser.add_argument('--native_agent_id_mode', type=str, default=None,
                    choices=['one_hot', 'learned'],
                    help='override model.native_agent_id_mode')
parser.add_argument('--native_actor_arch', type=str, default=None, choices=['mlp', 'iru'],
                    help='override model.native_actor_arch')
parser.add_argument('--native_value_arch', type=str, default=None, choices=['mlp', 'iru'],
                    help='override model.native_value_arch')
parser.add_argument('--iru_steps', type=int, default=None,
                    help='override both model.iru_actor_steps and model.iru_value_steps')
parser.add_argument('--iru_actor_steps', type=int, default=None,
                    help='override model.iru_actor_steps')
parser.add_argument('--iru_value_steps', type=int, default=None,
                    help='override model.iru_value_steps')
parser.add_argument('--iru_hidden_dim', type=int, default=None,
                    help='override model.iru_hidden_dim')
parser.add_argument('--iru_num_blocks', type=int, default=None,
                    help='override model.iru_num_blocks')
parser.add_argument('--profile_performance', type=str2bool, nargs='?', const=True, default=None,
                    help='record parameter count, latency, update time, and peak GPU memory')
parser.add_argument('--episodes', type=int, default=None,
                    help='override trainer.episodes (the final episode boundary)')
parser.add_argument('--resume_episode', type=int, default=None,
                    help='load model/<episode>_<rank>.pt before training and continue from this episode')
parser.add_argument('--config_snapshot', type=str, default=None,
                    help='rebuild model/trainer/logger/world settings from a saved hyperparameters.json')
parser.add_argument('--hyper_residual', type=str2bool, nargs='?', const=True, default=None,
                    help='override model.hyper_residual')
parser.add_argument('--hyper_residual_mode', type=str, default=None,
                    choices=['full', 'lora', 'low_rank', 'low-rank', 'head', 'head_only', 'head-only',
                             'last_layer', 'last-layer'],
                    help='override model.hyper_residual_mode')
parser.add_argument('--hyper_residual_scale', type=float, default=None,
                    help='override model.hyper_residual_scale')
parser.add_argument('--hyper_residual_actor_scale', type=float, default=None,
                    help='override model.hyper_residual_actor_scale')
parser.add_argument('--hyper_residual_value_scale', type=float, default=None,
                    help='override model.hyper_residual_value_scale')
parser.add_argument('--hyper_lora_rank', type=int, default=None,
                    help='override model.hyper_lora_rank')
parser.add_argument('--hyper_lora_actor_rank', type=int, default=None,
                    help='override model.hyper_lora_actor_rank')
parser.add_argument('--hyper_lora_value_rank', type=int, default=None,
                    help='override model.hyper_lora_value_rank')
parser.add_argument('--hyper_lora_bias', type=str2bool, nargs='?', const=True, default=None,
                    help='override model.hyper_lora_bias')
parser.add_argument('--hyper_head_mode', type=str, default=None,
                    choices=['flat', 'layerwise', 'chunked'],
                    help='override model.hyper_head_mode')
parser.add_argument('--hyper_chunk_size', type=int, default=None,
                    help='override model.hyper_chunk_size (chunked head only)')
parser.add_argument('--hyper_chunk_embed_dim', type=int, default=None,
                    help='override model.hyper_chunk_embed_dim (chunked head only)')
parser.add_argument('--save_rate', type=int, default=None,
                    help='override logger.save_rate (checkpoint every N episodes)')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

logging_level = logging.INFO
if args.debug:
    logging_level = logging.DEBUG


class Runner:
    def __init__(self, pArgs):
        """
        instantiate runner object with processed config and register config into Registry class
        """
        self.config, self.duplicate_config = build_config(pArgs)
        self.config_registry()

    def config_registry(self):
        """
        Register config into Registry class
        """

        interface.Command_Setting_Interface(self.config)
        interface.Logger_param_Interface(self.config)  # register logger path
        interface.World_param_Interface(self.config)
        if self.config['model'].get('graphic', False):
            param = Registry.mapping['world_mapping']['setting'].param
            if self.config['command']['world'] in ['cityflow', 'sumo']:
                roadnet_path = param['dir'] + param['roadnetFile']
            else:
                roadnet_path = param['road_file_addr']
            interface.Graph_World_Interface(roadnet_path)  # register graphic parameters in Registry class
        interface.Logger_path_Interface(self.config)
        # make output dir if not exist
        if not os.path.exists(Registry.mapping['logger_mapping']['path'].path):
            os.makedirs(Registry.mapping['logger_mapping']['path'].path)        
        interface.Trainer_param_Interface(self.config)
        interface.ModelAgent_param_Interface(self.config)

    def run(self):
        logger = setup_logging(logging_level)
        self.trainer = Registry.mapping['trainer_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](logger)
        self.task = Registry.mapping['task_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](self.trainer)
        start_time = time.time()
        self.task.run()
        logger.info(f"Total time taken: {time.time() - start_time}")


if __name__ == '__main__':
    test = Runner(args)
    test.run()

# python run.py --task tsc --agent adapt_comm --world cityflow --network cityflow_grid4x4 --prefix my_adapt_exp
