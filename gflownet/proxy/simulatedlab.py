# the idea is that we fit a lightgbm regressor for the lab data
import lightgbm as lgb
import torch

from gflownet.proxy.base import Proxy


class SimulatedLab(Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = lgb.Booster(model_file="lightgbm_model.txt")


    # def get_next(self, x: torch.Tensor):
    #     return self.model(x)
    
    def __call__(self, x):
        res = self.model.predict(x)
        print("res", res)
        return res
