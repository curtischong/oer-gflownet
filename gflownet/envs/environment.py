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

from gflownet.envs.base import GFlowNetEnv
import numpy as np

NUM_ELEMENTS = 5


class OEREnv(GFlowNetEnv):
    def __init__(self):
        super().__init__()
        self.state = np.zeros(NUM_ELEMENTS)
        self.source = np.zeros(NUM_ELEMENTS)

    # for continuous actions: https://github.com/alexhernandezgarcia/gflownet/blob/main/gflownet/envs/cube.py#L337
    def get_action_space(self):
        self.eos = tuple([np.inf] * NUM_ELEMENTS)
        self.representative_action = tuple([0.0] * NUM_ELEMENTS)
        return [self.representative_action, self.eos]

    def reset(self):
        self.state = np.zeros(NUM_ELEMENTS)
