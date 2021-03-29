from typing import List
from hypersp.utils.helpers import add_ctc_labels, print_dict
from hypersp.models import JasperEncoderDecoder
from hypersp.modules import AudioPreprocessing, CTCDecoder
from hypersp.datasets.features import audio_from_file
import os
import sys
import torch
import torch.nn as nn
import time
import librosa
import yaml

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


class Recognizer:

    def __init__(self,
                 beam_size=5,
                 model_cfg='',
                 ckpt='',
                 max_duration=None,
                 pad_to=None,
                 cpu=False,
                 amp=True,
                 output_dir='./results/',
                 task_path="",
                 vocab="",
                 ):
        self.beam_size = beam_size
        self.model_cfg = model_cfg
        self.ckpt = ckpt
        self.max_duration = max_duration
        self.pad_to = pad_to
        self.cpu = cpu
        self.amp = amp
        self.output_dir = output_dir
        self.task_path = task_path
        self.vocab = vocab

    def decode(self, prob_tensor, pred_tensor, labels, offset=30):
        """
        Takes output of greedy ctc decoder and performs ctc decoding algorithm to
        remove duplicates and special symbol. Returns prediction
        Args:
            tensor: model output tensor
            label: A list of labels
        Returns:
            prediction
        """
        blank_id = len(labels) - 1
        hypotheses = []
        labels_map = dict([(i, labels[i]) for i in range(len(labels))])
        batched_decoded = []

        # iterate over batch
        for ind in range(prob_tensor.shape[0]):
            prediction = pred_tensor[ind].numpy().tolist()
            prob = prob_tensor[ind].numpy().tolist()
            # CTC decoding procedure
            previous = blank_id  # id of a blank symbol
            decoded = []

            for idx, p in enumerate(prediction):
                if p[0] != previous and p[0] != blank_id:
                    if len(p) > 1 and p[1] == blank_id and prob[idx][1] > -1:
                        continue

                    if len(decoded) > 0 and idx - decoded[-1][1] > offset:
                        batched_decoded.append(decoded)
                        decoded = []
                    decoded.append([ind, idx, [labels_map[c]
                                               for c in p], prob[idx]])

                previous = p[0]

            batched_decoded.append(decoded)

        return batched_decoded

    def sec2time(self, sec, n_msec=3):
        ''' Convert seconds to 'D days, HH:MM:SS.FFF' '''
        if hasattr(sec, '__len__'):
            return [self.sec2time(s) for s in sec]
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        if n_msec > 0:
            pattern = '%%02d:%%02d:%%0%d.%df' % (n_msec+3, n_msec)
        else:
            pattern = r'%02d:%02d:%02d'
        if d == 0:
            return pattern % (h, m, s)
        return ('%d days, ' + pattern) % (d, h, m, s)

    def transcribe(
            self,
            wav,
            option=0):
        """performs inference / evaluation
        Args:
            audio_processor: data processing module
            encoderdecoder: acoustic model
            greedy_decoder: greedy decoder
            labels: list of labels as output vocabularyㄴ
            args: script input arguments
        """
        with torch.no_grad():
            if wav:
                audio, audio_len = audio_from_file(
                    wav, target_sr=16000, device=self.device)
                t3 = time.perf_counter()
                features, lens = self.audio_preprocessor(audio, audio_len)
                if not self.device.type == 'cpu':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                if not self.device.type == 'cpu' and self.amp:
                    with torch.cuda.amp.autocast():
                        t_log_probs_e, _ = self.encoderdecoder.infer(
                            (features, lens))
                else:
                    t_log_probs_e, _ = self.encoderdecoder.infer(
                        (features, lens))
                t2 = time.perf_counter()
                prob_tensor, pred_tensor = self.ctc_decoder(
                    log_probs=t_log_probs_e,  beam_size=self.beam_size)
                if not self.device.type == 'cpu':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                prob_cpu_tensor = prob_tensor.float().cpu()
                pred_cpu_tensor = pred_tensor.long().cpu()

                os.makedirs(self.output_dir, exist_ok=True)

                text = []

                if option == 0:
                    # (batch, index, beam_pred, beam_prob)
                    batched_decoded = self.decode(
                        prob_cpu_tensor, pred_cpu_tensor, self.labels, offset=35)

                    sentence_index = 1

                    for ind in range(len(batched_decoded)):
                        tmp = []
                        for val in batched_decoded[ind]:
                            tmp.append(val[2][0])
                        text.append(f"{sentence_index}\n")
                        text.append(
                            f"{self.sec2time(round(batched_decoded[ind][0][1] * 0.02, 3))} --> {self.sec2time(round((batched_decoded[ind][-1][1] + 3) * 0.02, 3))}\n{''.join(tmp).strip()}\n\n")
                        sentence_index += 1

                elif option == 1:
                    # (batch, index, beam_pred, beam_prob)
                    batched_decoded = self.decode(
                        prob_cpu_tensor, pred_cpu_tensor, self.labels, offset=70)

                    for ind in range(len(batched_decoded)):
                        tmp = []
                        for val in batched_decoded[ind]:
                            tmp.append(val[2][0])
                        text.append(f"{''.join(tmp).strip()}\n\n")

                print("PREPROCESS TIME\t\t: {} ms".format((t0-t3)*1000.0))
                print("INFERENCE TIME\t\t: {} ms".format((t2-t0)*1000.0))
                print("DECODE TIME\t\t: {} ms".format((t1-t2)*1000.0))

                return ''.join(text)

    def load_model(self):

        if self.cpu:
            self.device = torch.device('cpu')
        else:
            assert(torch.cuda.is_available())
            self.device = torch.device('cuda')

        with open(self.model_cfg) as f:
            model_definition = yaml.load(f, Loader=yaml.FullLoader)

        dataset_vocab = []
        with open(self.vocab, "r", encoding="utf-8") as f:
            for line in f:
                token = line.split(' ')[0]
                if token == '':
                    dataset_vocab.append(' ')
                else:
                    dataset_vocab.append(token)
        dataset_vocab = sorted(dataset_vocab)
        self.labels = add_ctc_labels(dataset_vocab)

        featurizer_config = model_definition['input_eval']
        featurizer_config["fp16"] = self.amp

        self.use_conv_mask = model_definition['encoder'].get('convmask', True)

        if self.max_duration is not None:
            featurizer_config['max_duration'] = self.max_duration
        if self.pad_to is not None:
            featurizer_config['pad_to'] = self.pad_to

        if featurizer_config['pad_to'] == "max":
            featurizer_config['pad_to'] = -1

        print('=== model_config ===')
        print_dict(model_definition)
        print()
        print('=== feature_config ===')
        print_dict(featurizer_config)
        print()
        self.audio_preprocessor = AudioPreprocessing(**featurizer_config)
        self.encoderdecoder = JasperEncoderDecoder(
            jasper_model_definition=model_definition, feat_in=1024, num_classes=len(self.labels))

        if self.ckpt is not None:
            print("loading model from ", self.ckpt)

            if os.path.isdir(self.ckpt):
                exit(0)
            else:
                checkpoint = torch.load(self.ckpt, map_location="cpu")
                sd = 'state_dict'
                for k in self.audio_preprocessor.state_dict().keys():
                    checkpoint[sd][k] = checkpoint[sd].pop(
                        "audio_preprocessor." + k)
                self.audio_preprocessor.load_state_dict(
                    checkpoint[sd], strict=False)
                self.encoderdecoder.load_state_dict(
                    checkpoint[sd], strict=False)

        print("audio_preprocessor.normalize: ",
              self.audio_preprocessor.featurizer.normalize)

        self.audio_preprocessor.to(self.device)
        self.encoderdecoder.to(self.device)
        self.ctc_decoder = CTCDecoder()

        self.audio_preprocessor.eval()
        self.encoderdecoder.eval()
        self.ctc_decoder.eval()


if __name__ == "__main__":
    recognizer = Recognizer()
    recognizer.load_model()
    recognizer.transcribe('test.wav')
