import argparse
import itertools
from typing import List
from tqdm import tqdm
import math
import yaml
import toml
from hypersp.datasets.dataset import AudioToTextDataLayer
from hypersp.utils.helpers import process_evaluation_batch, process_evaluation_epoch, add_ctc_labels, print_dict, model_multi_gpu, __ctc_decoder_predictions_tensor
from hypersp.models import JasperEncoderDecoder
from hypersp.modules import AudioPreprocessing, GreedyCTCDecoder
from hypersp.datasets.features import audio_from_file
import torch
import torch.nn as nn
import apex
from apex import amp
import random
import numpy as np
import pickle
import time
import os
import librosa


def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')

    parser.register("type", "bool", lambda x: x.lower()
                    in ("yes", "true", "t", "1"))

    parser.add_argument("--local_rank", default=None, type=int)
    parser.add_argument("--batch_size", default=16,
                        type=int, help='data batch size')
    parser.add_argument("--steps", default=None,
                        help='if not specified do evaluation on full dataset. otherwise only evaluates the specified number of iterations for each worker', type=int)
    parser.add_argument("--model_cfg", type=str,
                        help='relative model configuration path given dataset folder')
    parser.add_argument("--dataset_dir", type=str,
                        help='absolute path to dataset folder')
    parser.add_argument("--val_manifest", type=str,
                        help='relative path to evaluation dataset manifest file')
    parser.add_argument("--ckpt", default=None, type=str,
                        required=True, help='path to model checkpoint')
    parser.add_argument("--max_duration", default=None, type=float,
                        help='maximum duration of sequences. if None uses attribute from model configuration file')
    parser.add_argument("--pad_to", default=None, type=int,
                        help="default is pad to value as specified in model configurations. if -1 pad to maximum duration. If > 0 pad batch to next multiple of value")
    parser.add_argument("--amp", "--fp16",
                        action='store_true', help='use half precision')
    parser.add_argument("--cudnn_benchmark",
                        action='store_true', help="enable cudnn benchmark")
    parser.add_argument("--save_prediction", type=str, default=None,
                        help="if specified saves predictions in text form at this location")
    parser.add_argument("--save_reference", default=None,
                        type=str, help="if specified saves references in text form at this location")
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--output_dir", default="results/", type=str,
                        help="Output directory to store exported models. Only used if --export_model is used")
    parser.add_argument("--export_model", action='store_true',
                        help="Exports the audio_featurizer, encoder and decoder using torch.jit to the output_dir")
    parser.add_argument("--wav", type=str,
                        help='absolute path to .wav file (16KHz)')
    parser.add_argument("--cpu", action="store_true",
                        help="Run inference on CPU")
    parser.add_argument("--ema", action="store_true",
                        help="If available, load EMA model weights")
    parser.add_argument("--task_path", type=str,
                        required=True, help="task path")
    parser.add_argument("--vocab", default="vocab",
                        type=str, help='vocab path')

    # FIXME Unused, but passed by Triton helper scripts
    parser.add_argument("--pyt_fp16", action='store_true',
                        help='use half precision')

    return parser.parse_args()


def calc_wer(data_layer, audio_processor,
             encoderdecoder, greedy_decoder,
             labels, args, device):

    encoderdecoder = encoderdecoder.module if hasattr(
        encoderdecoder, 'module') else encoderdecoder
    with torch.no_grad():
        # reset global_var_dict - results of evaluation will be stored there
        _global_var_dict = {
            'predictions': [],
            'transcripts': [],
            'logits': [],
        }

        # Evaluation mini-batch for loop
        for it, data in enumerate(tqdm(data_layer.data_iterator)):

            tensors = [t.to(device) for t in data]

            t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = tensors

            t_processed_signal = audio_processor(
                t_audio_signal_e, t_a_sig_length_e)
            t_log_probs_e, _ = encoderdecoder.infer(t_processed_signal)
            t_predictions_e = greedy_decoder(t_log_probs_e)

            values_dict = dict(
                predictions=[t_predictions_e],
                transcript=[t_transcript_e],
                transcript_length=[t_transcript_len_e],
                # output=[t_log_probs_e] #TODO FIX MEMORY LEAK ISSUE
            )
            # values_dict will contain results from all workers

            process_evaluation_batch(
                values_dict, _global_var_dict, labels=labels)

            if args.steps is not None and it + 1 >= args.steps:
                break

        # final aggregation (over minibatches) and logging of results
        cer, _ = process_evaluation_epoch(_global_var_dict, use_cer=True)
        wer, _ = process_evaluation_epoch(_global_var_dict)

        return cer, wer, _global_var_dict


def run_once(audio_processor, encoderdecoder, greedy_decoder, audio, audio_len, labels, device, type='greedy'):
    features, lens = audio_processor(audio, audio_len)
    if not device.type == 'cpu':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    # TorchScripted model does not support (features, lengths)
    if isinstance(encoderdecoder, torch.jit.TracedModule):
        t_log_probs_e = encoderdecoder(features)
    else:
        t_log_probs_e, _ = encoderdecoder.infer((features, lens))
    if not device.type == 'cpu':
        torch.cuda.synchronize()

    if type == 'greedy':
        t_predictions_e = greedy_decoder(log_probs=t_log_probs_e)
        hypotheses = __ctc_decoder_predictions_tensor(
            t_predictions_e, labels=labels)

    t1 = time.perf_counter()

    print("INFERENCE TIME\t\t: {} ms".format((t1-t0)*1000.0))
    print("TRANSCRIPT\t\t:", hypotheses[0])


def eval(
        data_layer,
        audio_processor,
        encoderdecoder,
        greedy_decoder,
        labels,
        multi_gpu,
        device,
        args):
    """performs inference / evaluation
    Args:
        data_layer: data layer object that holds data loader
        audio_processor: data processing module
        encoderdecoder: acoustic model
        greedy_decoder: greedy decoder
        labels: list of labels as output vocabulary
        multi_gpu: true if using multiple gpus
        args: script input arguments
    """

    with torch.no_grad():
        if args.wav:
            audio, audio_len = audio_from_file(args.wav)
            run_once(audio_processor, encoderdecoder, greedy_decoder,
                     audio, audio_len, labels, device)
            return
        cer, wer, _global_var_dict = calc_wer(
            data_layer, audio_processor, encoderdecoder, greedy_decoder, labels, args, device)
        if (not multi_gpu or (multi_gpu and torch.distributed.get_rank() == 0)):
            print(f"==========>>>>>>Evaluation CER: {round(cer * 100, 2)}%")
            print(f"==========>>>>>>Evaluation WER: {round(wer * 100, 2)}%")

            if args.save_prediction is not None:
                with open(args.save_prediction, 'w') as fp:
                    fp.write('\n'.join(_global_var_dict['predictions']))
            if args.save_reference is not None:
                with open(args.save_reference, 'w') as fp:
                    fp.write('\n'.join(_global_var_dict['transcripts']))


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    multi_gpu = args.local_rank is not None

    if args.cpu:
        assert(not multi_gpu)
        device = torch.device('cpu')
    else:
        assert(torch.cuda.is_available())
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        print("CUDNN BENCHMARK ", args.cudnn_benchmark)

        if multi_gpu:
            print("DISTRIBUTED with ", torch.distributed.get_world_size())
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend='nccl', init_method='env://')

    optim_level = 3 if args.amp else 0

    with open(args.model_cfg) as f:
        model_definition = yaml.load(f, Loader=yaml.FullLoader)

    dataset_vocab = []
    with open(os.path.join(args.dataset_dir, args.vocab), "r", encoding="utf-8") as f:
        for line in f:
            token = line.split(' ')[0]
            if token == '':
                dataset_vocab.append(' ')
            else:
                dataset_vocab.append(token)

    dataset_vocab = sorted(dataset_vocab)

    ctc_vocab = add_ctc_labels(dataset_vocab)

    val_manifest = args.val_manifest
    featurizer_config = model_definition['input_eval']
    featurizer_config["optimization_level"] = optim_level
    featurizer_config["fp16"] = args.amp

    args.use_conv_mask = model_definition['encoder'].get('convmask', True)
    if args.use_conv_mask and args.export_model:
        print('WARNING: Masked convs currently not supported for TorchScript. Disabling.')
        model_definition['encoder']['convmask'] = False

    if args.max_duration is not None:
        featurizer_config['max_duration'] = args.max_duration
    if args.pad_to is not None:
        featurizer_config['pad_to'] = args.pad_to

    if featurizer_config['pad_to'] == "max":
        featurizer_config['pad_to'] = -1

    print('=== model_config ===')
    print_dict(model_definition)
    print()
    print('=== feature_config ===')
    print_dict(featurizer_config)
    print()
    data_layer = None

    if args.wav is None:
        data_layer = AudioToTextDataLayer(
            task_path=args.task_path,
            dataset_dir=args.dataset_dir,
            featurizer_config=featurizer_config,
            manifest_filepath=val_manifest,
            labels=dataset_vocab,
            batch_size=args.batch_size,
            pad_to_max=featurizer_config['pad_to'] == -1,
            shuffle=False,
            multi_gpu=multi_gpu)
    audio_preprocessor = AudioPreprocessing(**featurizer_config)

    if model_definition["model"] == "Jasper":
        encoderdecoder = JasperEncoderDecoder(
            jasper_model_definition=model_definition, feat_in=1024, num_classes=len(ctc_vocab))

    if args.ckpt is not None:
        print("loading model from ", args.ckpt)

        if os.path.isdir(args.ckpt):
            exit(0)
        else:
            checkpoint = torch.load(args.ckpt, map_location="cpu")
            if args.ema and 'ema_state_dict' in checkpoint:
                print('Loading EMA state dict')
                sd = 'ema_state_dict'
            else:
                sd = 'state_dict'

            for k in audio_preprocessor.state_dict().keys():
                checkpoint[sd][k] = checkpoint[sd].pop(
                    "audio_preprocessor." + k)
            audio_preprocessor.load_state_dict(checkpoint[sd], strict=False)
            encoderdecoder.load_state_dict(checkpoint[sd], strict=False)

    greedy_decoder = GreedyCTCDecoder()

    if args.wav is None:
        N = len(data_layer)
        step_per_epoch = math.ceil(N / (args.batch_size * (
            1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())))

        if args.steps is not None:
            print('-----------------')
            print('Have {0} examples to eval on.'.format(args.steps * args.batch_size * (
                1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())))
            print('Have {0} steps / (gpu * epoch).'.format(args.steps))
            print('-----------------')
        else:
            print('-----------------')
            print('Have {0} examples to eval on.'.format(N))
            print('Have {0} steps / (gpu * epoch).'.format(step_per_epoch))
            print('-----------------')

    print("audio_preprocessor.normalize: ",
          audio_preprocessor.featurizer.normalize)

    audio_preprocessor.to(device)
    encoderdecoder.to(device)

    if args.amp:
        encoderdecoder = amp.initialize(models=encoderdecoder,
                                        opt_level='O'+str(optim_level))

    encoderdecoder = model_multi_gpu(encoderdecoder, multi_gpu)
    audio_preprocessor.eval()
    encoderdecoder.eval()
    greedy_decoder.eval()

    eval(
        data_layer=data_layer,
        audio_processor=audio_preprocessor,
        encoderdecoder=encoderdecoder,
        greedy_decoder=greedy_decoder,
        labels=ctc_vocab,
        args=args,
        device=device,
        multi_gpu=multi_gpu)


if __name__ == "__main__":
    args = parse_args()

    print_dict(vars(args))

    main(args)
