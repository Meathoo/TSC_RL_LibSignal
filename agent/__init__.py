from .base import BaseAgent
from .rl_agent import RLAgent
from .maxpressure import MaxPressureAgent
from .colight import CoLightAgent
from .dqn import DQNAgent
from .sotl import SOTLAgent
from .frap import FRAP_DQNAgent
from .ppo_pfrl import IPPO_pfrl
from .maddpg_v2 import MADDPGAgent
from .magd import MAGDAgent
from .presslight import PressLightAgent
from .fixedtime import FixedTimeAgent
from .mplight import MPLightAgent
from .h2tsc_agent import H2TSCAgent
# Legacy TD3/MB-HyperLight is parked while the paper-faithful PPO/MAPPO
# branches are the active HyperMARL implementations.
# from .hyperlight import HyperLightAgent
from .hyperlight_ppo import HyperLightPPOAgent, HyperLightMAPPOAgent
from .native_ppo import NativePPOAgent, NativeMAPPOAgent

try:
    from .adapt_comm_agent import ADAPTCommAgent
except ImportError:
    ADAPTCommAgent = None
