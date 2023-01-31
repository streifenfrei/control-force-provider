from .basics import *
from .dqn import *
from .monte_carlo import *
from .hierarchical_rl import *

context_mapping = {
    "dqn": DQNContext,
    "dqn_naf": DQNNAFContext,
    "mc": MonteCarloContext,
    "hrl": HierarchicalRLContext
}
