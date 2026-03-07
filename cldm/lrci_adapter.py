import torch
import torch.nn as nn

class LRCIAdapter(nn.Module):

    def __init__(self, dim, rank=4):

        super().__init__()

        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)

    def forward(self, x):

        return self.B(self.A(x))
