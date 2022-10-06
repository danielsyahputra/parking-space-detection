import torch
import torch.nn.functional as F

from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self, in_features, out_features=1024) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.fc3 = nn.Linear(out_features, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x