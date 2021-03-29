import torch
import torch.nn as nn
from hypersp.modules import AudioPreprocessing, SpectrogramAugmentation
from hypersp.modules import ConvDecoderForCTC, JasperEncoder
import random


class JasperEncoderDecoder(nn.Module):
    """Contains jasper encoder and decoder
    """

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.transpose_in = kwargs.get("transpose_in", False)

        self.encoder = JasperEncoder(**kwargs.get("jasper_model_definition"))
        self.decoder = ConvDecoderForCTC(feat_in=kwargs.get("feat_in"),
                                         num_classes=kwargs.get("num_classes"))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if self.encoder.use_conv_mask:
            t_encoded_t, t_encoded_len_t = self.encoder(x)
        else:
            if self.transpose_in:
                x = x.transpose(1, 2)
            t_encoded_t = self.encoder(x)

        out = self.decoder(t_encoded_t)
        if self.encoder.use_conv_mask:
            return out, t_encoded_len_t
        else:
            return out

    def infer(self, x):
        if self.encoder.use_conv_mask:
            return self.forward(x)
        else:
            ret = self.forward(x[0])
            return ret, len(ret)


class Jasper(JasperEncoderDecoder):
    """Contains data preprocessing, spectrogram augmentation, jasper encoder and decoder
    """

    def __init__(self, **kwargs):
        JasperEncoderDecoder.__init__(self, **kwargs)
        feature_config = kwargs.get("feature_config")
        if self.transpose_in:
            feature_config["transpose"] = True
        self.audio_preprocessor = AudioPreprocessing(**feature_config)
        self.data_spectr_augmentation = SpectrogramAugmentation(
            **feature_config)
