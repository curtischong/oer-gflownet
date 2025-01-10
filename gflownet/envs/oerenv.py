# the goal of the model is to specify what percentage of each element to use and minimize the overpotential of the catalyst
# GFlowNets is the agent that searches the space (performs a similar task as BO)
# how might I create a model that simluates the lab?
# I can just use a simple tree-based model for now. maybe then train a linear model to get predictions
# I can sort the data from worse to best overpotential. then give the gflownet badoverpotentials at first, and ohpe that it discovers something on the pareto front? something with a good overpotential?
# questions:
# what is the reward function? - ans: It's the simulated lab I wrote


# action space: change the percentages of all 5 elements
# state space: it's the percentages of all 5 elements

# different action space? choose one element percentage one at a time
# state space: the "trajectory" of elements chosen. but this doesnt' make sense. the order in which percentages don't count
# it's just kinda weird having the state space be just 1 state (not like a markov process) -> since it doesn't really motivate the need for a gflow net. but whatever
# I actually think it's fine that we're using gflow nets since think of this as just a tree with height 1. you still want to sample states proportional to the reward

# I don't htink that gflownets can handle continuous actions?

from typing import List, Optional, Tuple, Union
from gflownet.envs.base import GFlowNetEnv
import numpy as np
import torch
from torchtyping import TensorType

from gflownet.envs.cube import ContinuousCube
from gflownet.proxy.uniform import Uniform

NUM_ELEMENTS = 6

# I'm referencing ContinuousTorus since it only has fully continuous actions
class OEREnv(ContinuousCube):
    def states2proxy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: clips the
        states into [0, 1] and maps them to [CELL_MIN, CELL_MAX]

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        # states = tfloat(states, device=self.device, float_type=self.float)
        # return 2.0 * torch.clip(states, min=0.0, max=CELL_MAX) - CELL_MAX
        return states