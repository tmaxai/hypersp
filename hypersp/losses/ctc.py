import torch
from torch import nn


class CTCLossNM:
    """ CTC loss
    """

    def __init__(self, **kwargs):
        self._blank = kwargs['num_classes'] - 1
        self._criterion = nn.CTCLoss(blank=self._blank, reduction='none')

    def __call__(self, log_probs, targets, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        targets = targets.long()
        loss = self._criterion(log_probs.transpose(1, 0), targets, input_length,
                               target_length)
        # note that this is different from reduction = 'mean'
        # because we are not dividing by target lengths
        return torch.mean(loss)
