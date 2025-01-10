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

from typing import List, Optional, Tuple
from gflownet.envs.base import GFlowNetEnv
import numpy as np
import torch
from torchtyping import TensorType

from gflownet.proxy.uniform import Uniform

NUM_ELEMENTS = 6

# I'm referencing ContinuousTorus since it only has fully continuous actions
class OEREnv(GFlowNetEnv):
    def __init__(self, **kwargs):
        self.state = np.zeros(NUM_ELEMENTS)
        self.source = np.zeros(NUM_ELEMENTS)
        # self.mask_dim = 2 # the dim is 2. 1 is for the action, 1 is for the eos
        super().__init__(**kwargs)

    # for continuous actions: https://github.com/alexhernandezgarcia/gflownet/blob/main/gflownet/envs/cube.py#L337
    def get_action_space(self):
        self.eos = tuple([np.inf] * NUM_ELEMENTS)
        self.representative_action = tuple([0.0] * NUM_ELEMENTS)
        return [self.representative_action, self.eos]

    # def reset(self):
    #     self.state = np.zeros(NUM_ELEMENTS)
    #     self.source = np.zeros(NUM_ELEMENTS)

    # def done(self):
    #     pass

    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        policy_output = torch.ones(
            NUM_ELEMENTS, dtype=self.float, device=self.device
        )
        # policy_output[1::3] = params["vonmises_mean"]
        # policy_output[2::3] = params["vonmises_concentration"]
        return policy_output

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple int values indicating the
            dimensions to increment by 1.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """

        # print("action", action)
        # If action is eos
        if action == self.eos:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is not eos, then perform action
        else:
            self.state = action
            valid = True
            return self.state, action, valid
        
    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
        states_from: Optional[List] = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs. The angle increments
        that form the actions are sampled from a mixture of Von Mises distributions.

        A distinction between forward and backward actions is made and specified by the
        argument is_backward, in order to account for the following special cases:

        Forward:

        - If the number of steps is equal to the maximum, then the only valid action is
          EOS.

        Backward:

        - If the number of steps is equal to 1, then the only valid action is to return
          to the source. The specific action depends on the current state.

        Args
        ----
        policy_outputs : tensor
            The output of the GFlowNet policy model.

        mask : tensor
            The mask containing information about special cases.

        states_from : tensor
            The states originating the actions, in GFlowNet format.

        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default).
        """
        # device = policy_outputs.device
        n_states = policy_outputs.shape[0]
        # logprobs = torch.zeros(
        #     (n_states, NUM_ELEMENTS), dtype=self.float, device=self.device
        # )
        # Initialize actions tensor with EOS actions (inf) since these will be the
        # actions for several special cases in both forward and backward actions.
        # actions_tensor = torch.full(
        #     (n_states, self.n_dim), torch.inf, dtype=self.float, device=device
        # )
        # Sample angle increments
        if sampling_method == "uniform":
            percentages = Uniform(
                torch.zeros(NUM_ELEMENTS),
                torch.ones(NUM_ELEMENTS),
            ).reshape(n_states, NUM_ELEMENTS)
        elif sampling_method == "policy":
            logits = policy_outputs
            percentages = torch.softmax(logits, dim=1) # curtis: this makes sense. we need to normalize between 0 and 1.
        # Catch special case for backwards backt-to-source (BTS) actions
        if is_backward:
            return torch.zeros(n_states, NUM_ELEMENTS), torch.zeros(NUM_ELEMENTS)
        # TODO: is this too inefficient because of the multiple data transfers?
        # print("percentages", percentages)
        return percentages, None # in base.py, the second param isn't even used
        
    # def get_mask_invalid_actions_forward(
    #     self,
    #     state: Optional[List[int]] = None,
    #     done: Optional[bool] = None,
    # ) -> List[bool]:
    #     """
    #     Returns a list of length the action space with values:
    #         - True if the forward action is invalid from the current state.
    #         - False otherwise.

    #     Args
    #     ----
    #     state : tensor
    #         Input state. If None, self.state is used.

    #     done : bool
    #         Whether the trajectory is done. If None, self.done is used.

    #     Returns
    #     -------
    #     A list of boolean values.
    #     """
    #     state = self._get_state(state)
    #     done = not (self.state == self.source).all()
    #     if done:
    #         mask = [True for _ in range(self.action_space_dim)]
    #         mask[self.action_space.index(self.eos)] = False # only make eos valid
    #     else:
    #         mask = [False for _ in range(self.action_space_dim)]
    #         mask[self.action_space.index(self.eos)] = True # make eos invalid
    #     return mask

    # implimenting our own get_logprobs since base.py says for continuous envs, we need to define our own
    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", "n_dim"],
        mask: TensorType["n_states", "1"],
        states_from: Optional[List] = None,
        is_backward: bool = False,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.

        Args
        ----
        policy_outputs : tensor
            The output of the GFlowNet policy model.

        mask : tensor
            The mask containing information special cases.

        actions : tensor
            The actions (angle increments) from each state in the batch for which to
            compute the log probability.

        states_from : tensor
            Ignored.

        is_backward : bool
            Ignored.
        """
        # device = policy_outputs.device
        # # do_sample = torch.all(~mask, dim=1)
        # n_states = policy_outputs.shape[0]
        # logprobs = torch.zeros(n_states, self.n_dim).to(device)
        # print("logprobs", logprobs)
        # if torch.any(do_sample):
        #     mix_logits = policy_outputs[do_sample, 0::3].reshape(
        #         -1, self.n_dim, self.n_comp
        #     )
        #     mix = Categorical(logits=mix_logits)
        #     locations = policy_outputs[do_sample, 1::3].reshape(
        #         -1, self.n_dim, self.n_comp
        #     )
        #     concentrations = policy_outputs[do_sample, 2::3].reshape(
        #         -1, self.n_dim, self.n_comp
        #     )
        #     vonmises = VonMises(
        #         locations,
        #         torch.exp(concentrations) + self.vonmises_min_concentration,
        #     )
        #     distr_angles = MixtureSameFamily(mix, vonmises)
        #     logprobs[do_sample] = distr_angles.log_prob(actions[do_sample])
        # logprobs = torch.sum(logprobs, axis=1)
        return torch.log(policy_outputs)

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        # return torch.tensor([False, False], dtype=torch.bool, device=self.device)
        return torch.zeros((NUM_ELEMENTS), dtype=torch.bool, device=self.device)

    def state2readable(self, state: List[int] = None) -> str:
        print(str(state))

    def get_max_traj_length(self):
        return 1
