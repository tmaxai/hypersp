import torch
from torch import nn

from hypersp.datasets.features import FeatureFactory


class AudioPreprocessing(nn.Module):
    """GPU accelerated audio preprocessing
    """
    __constants__ = ["optim_level"]

    def __init__(self, **kwargs):
        nn.Module.__init__(self)    # For PyTorch API
        self.optim_level = kwargs.get('optimization_level', 0)
        self.featurizer = FeatureFactory.from_config(kwargs)
        self.transpose_out = kwargs.get("transpose_out", False)

    @torch.no_grad()
    def forward(self, input_signal, length):
        processed_signal = self.featurizer(input_signal, length)
        processed_length = self.featurizer.get_seq_len(length)
        if self.transpose_out:
            processed_signal.transpose_(2, 1)
            return processed_signal, processed_length
        else:
            return processed_signal, processed_length
