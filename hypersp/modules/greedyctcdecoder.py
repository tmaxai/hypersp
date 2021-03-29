import torch
from torch import nn


class GreedyCTCDecoder(nn.Module):
    """ Greedy CTC Decoder
    """

    def __init__(self, **kwargs):
        nn.Module.__init__(self)    # For PyTorch API

    @torch.no_grad()
    def forward(self, log_probs):
        argmx = log_probs.argmax(dim=-1, keepdim=False).int()
        return argmx
