import torch
import copy
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.container import ModuleList

class NeighborAttention(nn.Module):
    def __init__(self, n, d_model) -> None:
        super().__init__()
        self.n = n
        self.linear = nn.Linear((n + 1) * d_model, d_model)