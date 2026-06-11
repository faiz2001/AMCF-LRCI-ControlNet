import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiControlFusion(nn.Module):
    def __init__(self, dim, num_controls=1):
        super().__init__()
        self.dim = dim
        # Use adaptive scoring — works for 1 or more controls
        self.scorer = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, controls):
        # controls: list of tensors, each (B, C, H, W)
        # All must have same C, H, W — guaranteed by our per-level design
        if len(controls) == 1:
            return controls[0]  # Single control — passthrough, no fusion needed

        scores = []
        for c in controls:
            s = self.scorer(c)          # (B, 1, H, W)
            s = s.mean(dim=[2, 3])      # (B, 1) — global average
            scores.append(s)

        scores  = torch.cat(scores, dim=1)   # (B, num_controls)
        weights = F.softmax(scores, dim=1)   # (B, num_controls)

        fused = torch.zeros_like(controls[0])
        for i, c in enumerate(controls):
            w = weights[:, i].view(-1, 1, 1, 1)
            fused = fused + w * c

        return fused
