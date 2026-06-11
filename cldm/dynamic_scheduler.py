import torch

class DynamicControlScheduler:
    def __init__(self, T=1000):
        self.T = T

    def weight(self, t):
        # t can be int, float, or scalar tensor
        if isinstance(t, torch.Tensor):
            t = t.float()
        else:
            t = torch.tensor(float(t))
        # Returns scalar — caller reshapes as needed
        return torch.sigmoid(-0.01 * (t - self.T / 2))
