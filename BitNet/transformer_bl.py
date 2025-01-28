import torch
import torch.nn as nn
import torch.nn.functional as F
from ffn import FeedForwardNetwork
from mha_bl import MultiHeadAttention
from bitlinear import BitLinear

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForwardNetwork(d_model, d_hidden)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output = self.attn(x)
        x = self.norm1(attn_output + x)

        ffn_out = self.ffn(x)
        x = self.norm2(ffn_out + x)
        return x

class BitNet(nn.Module):
    def __init__(self, d_model, d_input, d_hidden, n_heads, n_layers):
        super().__init__()
        self.emb = nn.Linear(d_input, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_hidden)
                for _ in range(n_layers)
            ]
        )

        self.out_layer = BitLinear(d_model, d_input)