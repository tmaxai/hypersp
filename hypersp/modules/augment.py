import torch
from torch import nn
import random


class SpectrogramAugmentation(nn.Module):
    """Spectrogram augmentation
    """

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.spec_cutout_regions = SpecCutoutRegions(kwargs)
        self.spec_augment = SpecAugment(kwargs)

    @torch.no_grad()
    def forward(self, input_spec):
        augmented_spec = self.spec_cutout_regions(input_spec)
        augmented_spec = self.spec_augment(augmented_spec)
        return augmented_spec


class SpecAugment(nn.Module):
    """Spec augment. refer to https://arxiv.org/abs/1904.08779
    """

    def __init__(self, cfg):
        super(SpecAugment, self).__init__()
        self.cutout_x_regions = cfg.get('cutout_x_regions', 0)
        self.cutout_y_regions = cfg.get('cutout_y_regions', 0)

        self.cutout_x_width = cfg.get('cutout_x_width', 10)
        self.cutout_y_width = cfg.get('cutout_y_width', 10)

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        mask = torch.zeros(x.shape, dtype=torch.bool)
        for idx in range(sh[0]):
            for _ in range(self.cutout_x_regions):
                cutout_x_left = int(random.uniform(
                    0, sh[1] - self.cutout_x_width))

                mask[idx, cutout_x_left:cutout_x_left +
                     self.cutout_x_width, :] = 1

            for _ in range(self.cutout_y_regions):
                cutout_y_left = int(random.uniform(
                    0, sh[2] - self.cutout_y_width))

                mask[idx, :, cutout_y_left:cutout_y_left + self.cutout_y_width] = 1

        x = x.masked_fill(mask.to(device=x.device), 0)

        return x


class SpecCutoutRegions(nn.Module):
    """Cutout. refer to https://arxiv.org/pdf/1708.04552.pdf
    """

    def __init__(self, cfg):
        super(SpecCutoutRegions, self).__init__()

        self.cutout_rect_regions = cfg.get('cutout_rect_regions', 0)
        self.cutout_rect_time = cfg.get('cutout_rect_time', 5)
        self.cutout_rect_freq = cfg.get('cutout_rect_freq', 20)

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        mask = torch.zeros(x.shape, dtype=torch.bool)

        for idx in range(sh[0]):
            for i in range(self.cutout_rect_regions):
                cutout_rect_x = int(random.uniform(
                    0, sh[1] - self.cutout_rect_freq))
                cutout_rect_y = int(random.uniform(
                    0, sh[2] - self.cutout_rect_time))

                w_x = int(random.uniform(0, self.cutout_rect_time))
                w_y = int(random.uniform(0, self.cutout_rect_freq))

                mask[idx,
                     cutout_rect_x: cutout_rect_x + w_x,
                     cutout_rect_y: cutout_rect_y + w_y] = 1

        x = x.masked_fill(mask.type(torch.bool).to(device=x.device), 0)

        return x
