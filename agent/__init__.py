from .base import BaseAgent
from .rl_agent import RLAgent
from .maxpressure import MaxPressureAgent
try:
    from .colight import CoLightAgent
except ImportError:
    CoLightAgent = None
try:
    from .dqn import DQNAgent
except ImportError:
    DQNAgent = None
from .sotl import SOTLAgent
try:
    from .frap import FRAP_DQNAgent
except ImportError:
    FRAP_DQNAgent = None
try:
    from .ppo_pfrl import IPPO_pfrl
except ImportError:
    IPPO_pfrl = None
from .maddpg_v2 import MADDPGAgent
from .magd import MAGDAgent
from .presslight import PressLightAgent
from .fixedtime import FixedTimeAgent
try:
    from .mplight import MPLightAgent
except ImportError:
    MPLightAgent = None
# Legacy TD3/MB-HyperLight is parked while the paper-faithful HyperMARL
# branches below stay free of surrogate dynamics additions.
# from .hyperlight import HyperLightAgent
from .hyperlight_ppo import (
    HyperLightPPOAgent,
    HyperLightMAPPOAgent,
    HyperLightGraphMAPPOAgent,
    HyperLightMAPPOCoSAgent,
    HyperLightMASPOAgent,
)
from .hyperlight_td3 import HyperLightTD3Agent, HyperLightMATD3Agent
from .native_ppo import NativePPOAgent, NativeMAPPOAgent
from .mat import MATAgent

try:
    from .adapt_comm_agent import ADAPTCommAgent
except ImportError:
    ADAPTCommAgent = None
