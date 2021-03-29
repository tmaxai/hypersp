import torch
from torch import nn


class CTCDecoder(nn.Module):
    """ CTC Decoder
    """

    def __init__(self, **kwargs):
        nn.Module.__init__(self)    # For PyTorch API

    @torch.no_grad()
    def forward(self, log_probs, beam_size):
        hyps = torch.topk(log_probs, beam_size)
        return hyps
