from .basics import *
from .dqn import *
from .monte_carlo import *
from .actor_critic import *
from .hierarchical_rl import *

context_mapping = {
    "dqn": DQNContext,
    "dqn_naf": DQNNAFContext,
    "mc": MonteCarloContext,
    "ac": A2CContext,
    "sac": SACContext,
    "hrl": HierarchicalRLContext
}
