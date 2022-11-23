from .basics import *
from .dqn import *
from .monte_carlo import *

context_mapping = {
    "dqn": DQNContext,
    "mc": MonteCarloContext
}
