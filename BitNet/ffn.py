import torch
import torch.nn as nn
import torch.nn.functional as F
from bitlinear import BitLinear

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()

        self.fc1 = BitLinear(d_model, d_hidden)
        self.fc2 = BitLinear(d_hidden, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(self.gelu(x))
        return x