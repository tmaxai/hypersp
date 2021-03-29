import torch
import torch.nn as nn
from hypersp.datasets.features import FeatureFactory
import random


def init_weights(m, mode='xavier_uniform'):
    if type(m) == nn.Conv1d or type(m) == MaskedConv1d:
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif type(m) == nn.BatchNorm1d:
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (kernel_size // 2) * dilation


activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}


def compute_new_kernel_size(kernel_size, kernel_width):
    new_kernel_size = max(int(kernel_size * kernel_width), 1)
    # If kernel is even shape, round up to make it odd
    if new_kernel_size % 2 == 0:
        new_kernel_size += 1
    return new_kernel_size


class MaskedConv1d(nn.Conv1d):
    """1D convolution with sequence masking
    """
    __constants__ = ["use_conv_mask"]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, use_conv_mask=True):
        super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride=stride,
                                           padding=padding, dilation=dilation,
                                           groups=groups, bias=bias)
        self.use_conv_mask = use_conv_mask

    def get_seq_len(self, lens):
        return ((lens + 2 * self.padding[0] - self.dilation[0] * (
            self.kernel_size[0] - 1) - 1) // self.stride[0] + 1)

    def forward(self, inp):
        if self.use_conv_mask:
            x, lens = inp
            max_len = x.size(2)
            idxs = torch.arange(max_len).to(lens.dtype).to(
                lens.device).expand(len(lens), max_len)
            mask = idxs >= lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
            del mask
            del idxs
            lens = self.get_seq_len(lens)
            return super(MaskedConv1d, self).forward(x), lens
