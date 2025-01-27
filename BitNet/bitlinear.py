import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Module):
    def __init__(self, d_input, d_output):
        super(BitLinear, self).__init__()
        self.input_dim = d_input
        self.output_dim = d_output

        self.weight = nn.Parameter(torch.randn(d_output, d_input))
        self.gamma = nn.Parameter(torch.ones(d_output)) # scaling factor (scalar)
        self.beta = nn.Parameter(torch.zeros(d_output)) # bias-like term (scalar)

        self.layernorm = nn.LayerNorm(d_input)

    def absmax_quantization(self, weights):
        absmax = weights.abs().max()
        return torch.sign(weights), absmax

    def forward(self, x):
        x = self.layernorm(x)

        quant_w, scaling_factor = self.absmax_quantization(self.weight)

        output = F.linear(x, quant_w)

        output = output * scaling_factor * self.gamma + self.beta
        return output