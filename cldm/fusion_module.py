import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiControlFusion(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.attn = nn.Linear(dim, 1)

    def forward(self, controls):

        scores = []

        for c in controls:
            scores.append(self.attn(c))

        scores = torch.cat(scores, dim=1)

        weights = F.softmax(scores, dim=1)

        fused = 0

        for i, c in enumerate(controls):
            fused += weights[:, i].unsqueeze(-1) * c

        return fused
