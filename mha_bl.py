import torch
import torch.nn as nn
import torch.nn.functional as F
from bitlinear import BitLinear

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads

        assert self.d_model % n_heads == 0, "n_heads must divide d_model"

        self.q_proj = BitLinear(d_model, d_model)
        self.k_proj = BitLinear(d_model, d_model)
        self.v_proj = BitLinear(d_model, d_model)
        self.out_proj = BitLinear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, d_model)
        
        return self.out_proj(context)