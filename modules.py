
import torch
import torch.nn as nn

class Concatenate(nn.Module):
    def __init__(self, axis, *args):
        super(Concatenate, self).__init__()
        self.axis = axis
    def forward(self, x1, x2):
        return torch.cat((x1, x2), self.axis)
