import argparse
import json
import os

from tqdm import tqdm
from typing import List


def main(args):

    vocab = dict()

    for fname in tqdm(args.input_json):
        with open(fname, "r") as fp:
            dataset = json.load(fp)
        for data in dataset:
            for key in data["transcript"]:
                try:
                    vocab[key] += 1
                except KeyError:
                    vocab[key] = 1

    if isinstance(args.additional_token, List):
        for key in args.additional_token:
            try:
                vocab[key] += 1
            except KeyError:
                vocab[key] = 1
    vocab["<unk>"] = 1

    with open(os.path.join(args.dest_dir, "vocab"), "w", encoding="utf-8") as fp:
        vocab = sorted(vocab.items(), key=(lambda x: x[0]), reverse=True)
        for k, v in vocab:
            fp.writelines(f"{k} {v}\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess LibriSpeech.')
    parser.add_argument('--input_json', nargs='+', type=str, required=True,
                        help='LibriSpeech collection input dir')
    parser.add_argument('--dest_dir', type=str, required=True,
                        help='Output dir')
    parser.add_argument('--additional_token', nargs='*', type=str,
                        help='additional token')
    args = parser.parse_args()

    args.dest_dir = args.dest_dir.rstrip('/')

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
