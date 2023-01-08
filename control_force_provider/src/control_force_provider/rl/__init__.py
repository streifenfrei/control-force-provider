from .basics import *
from .dqn import *
from .monte_carlo import *

context_mapping = {
    "dqn_naf": DQNNAFContext,
    "mc": MonteCarloContext
}
