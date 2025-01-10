# the idea is that we fit a lightgbm regressor for the lab data
import lightgbm as lgb
import torch

from gflownet.proxy.base import Proxy


class SimulatedLab(Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = lgb.Booster(model_file="lightgbm_model.txt")
        self._optimum = 0


    # def get_next(self, x: torch.Tensor):
    #     return self.model(x)
    
    def __call__(self, x):
        x = torch.tensor(x)
        print("x", x)

        # why is their documentation lying? -0.9980 is not within [0, 1]

        res = self.model.predict(torch.nn.functional.softmax(x, dim=-1))
        print("res", res)
        return -torch.tensor(res, dtype=torch.float) # make rewards negative. so a lower overenergy is better
