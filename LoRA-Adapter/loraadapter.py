import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRAAdapter(nn.Module):
    def __init__(self, s_input, s_output, rank=4, scaling_factor=0.1):
        super().__init__()

        self.a = nn.Parameter(torch.randn(s_input, rank))
        self.b = nn.Parameter(torch.randn(rank, s_output))
        self.alpha = scaling_factor

        self.adapter_layer = nn.Sequential(
            nn.Linear(s_input, s_output),
            nn.ReLU(),
            nn.Linear(s_output, s_output)
        )

    def forward(self, x):
        delta_w = torch.matmul(self.a, self.b)
        x = x + (delta_w * self.alpha)
        x = self.adapter_layer(x)
        return x
    

class LoRA_model(nn.Module):
    def __init__(self, s_input, s_output, s_hidden, rank=4):
        super().__init__()

        self.fc1 = nn.Linear(s_input, s_hidden)
        self.fc2 = nn.Linear(s_hidden, s_output)

        self.lora_adapter1 = LoRAAdapter(s_input, s_hidden, rank)
        self.lora_adapter2 = LoRAAdapter(s_hidden, s_output, rank)

    def forward(self, x):
        x = self.fc1(x)
        x = self.lora_adapter1(x)
        x = self.fc2(x)
        x = self.lora_adapter2(x)

        return x