import torch

class DynamicControlScheduler:

    def __init__(self, T=1000):

        self.T = T

    def weight(self, t):

        return torch.sigmoid(-0.01 * (t - self.T/2))
