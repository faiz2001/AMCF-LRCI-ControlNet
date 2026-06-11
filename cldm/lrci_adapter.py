import torch
import torch.nn as nn

class LRCIAdapter(nn.Module):
    def __init__(self, dim, rank=4):
        super().__init__()
        # Conv2d-based LoRA — works on (B, C, H, W) directly
        self.A = nn.Conv2d(dim, rank, kernel_size=1, bias=False)
        self.B = nn.Conv2d(rank, dim, kernel_size=1, bias=False)

        # Critical: init B to zero so adapter starts as identity
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        # x: (B, C, H, W)
        return x + self.B(self.A(x))   # residual
