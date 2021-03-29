from torch import nn
from hypersp.utils.common import init_weights


class ConvDecoderForCTC(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self._feat_in = kwargs.get("feat_in")
        self._num_classes = kwargs.get("num_classes")
        init_mode = kwargs.get('init_mode', 'xavier_uniform')

        self.decoder_layers = nn.Sequential(
            nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True),)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, encoder_output):
        out = self.decoder_layers(encoder_output[-1]).transpose(1, 2)
        return nn.functional.log_softmax(out, dim=2)
