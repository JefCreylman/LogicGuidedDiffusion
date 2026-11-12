import math
import torch
from torch import nn

LOG2 = math.log(2.0)

class FuzzyLogic(nn.Module):
    @staticmethod
    def log_not(x):
        return torch.log(1-torch.expm1(x))
    # def log_not(x):
    #     return torch.where(
    #         x < -LOG2,
    #         torch.log1p(-torch.exp(x)),
    #         torch.log(-torch.expm1(x))
    #     )

    @staticmethod
    def log_and(x, y):
        return x + y

    @staticmethod
    def log_or(x, y):
        return torch.logaddexp(x, FuzzyLogic.log_not(x) + y)

