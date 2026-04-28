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

try:
    from .adapt_comm_agent import ADAPTCommAgent
except ImportError:
    ADAPTCommAgent = None
