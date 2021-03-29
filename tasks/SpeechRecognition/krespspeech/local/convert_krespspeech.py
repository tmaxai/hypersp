import argparse
import os
import glob
import multiprocessing
import json
import re

import pandas as pd

from preprocessing_utils import parallel_preprocess
from cleaners import sentence_filter
from tqdm import tqdm


def __build_input_arr(input_dir):
    txt_files = glob.glob(os.path.join(input_dir, '**', '*.txt'),
                          recursive=True)
    input_data = []

    for txt_file in tqdm(txt_files):
        rel_path = os.path.relpath(txt_file, input_dir)
        with open(txt_file, encoding='utf-8') as fp:
            for line in fp:
                fname = os.path.basename(fp.name).split('.')[0]
                transcript = sentence_filter(line.strip())

                tmp = transcript.replace("<unk>", "")
                non_hangul = re.compile('[ 가-힣]+').sub("", tmp)
                if len(non_hangul) > 0:
                    continue
                if len(tmp) < 5:
                    continue

                input_data.append(dict(input_relpath=os.path.dirname(rel_path),
                                       input_fname=fname+'.wav',
                                       transcript=transcript))
    return input_data


def main(args):

    print("[%s] Scaning input dir..." % args.output_json)
    dataset = __build_input_arr(input_dir=args.input_dir)

    print("[%s] Converting audio files..." % args.output_json)
    prepset = parallel_preprocess(dataset=dataset,
                                  input_dir=args.input_dir,
                                  dest_dir=args.dest_dir,
                                  target_sr=args.target_sr,
                                  speed=args.speed,
                                  overwrite=args.overwrite,
                                  parallel=args.parallel)
    print("[%s] length of preprocessed dataset: %d" %
          (args.output_json, len(prepset)))
    dataset = [x for x in prepset if x is not False]
    print("[%s] length of dataset after removing false: %d" %
          (args.output_json, len(dataset)))

    print("[%s] Generating json..." % args.output_json)
    df = pd.DataFrame(dataset, dtype=object)

    # Save json with python. df.to_json() produces back slashed in file paths
    dataset = df.to_dict(orient='records')
    with open(args.output_json, 'w') as fp:
        json.dump(dataset, fp, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess ksponspeech.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='ksponspeech collection input dir')
    parser.add_argument('--dest_dir', type=str, required=True,
                        help='Output dir')
    parser.add_argument('--output_json', type=str, default='./',
                        help='name of the output json file.')
    parser.add_argument('-s', '--speed', type=float, nargs='*',
                        help='Speed perturbation ratio')
    parser.add_argument('--target_sr', type=int, default=None,
                        help='Target sample rate. '
                        'defaults to the input sample rate')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite file if exists')
    parser.add_argument('--parallel', type=int, default=round(multiprocessing.cpu_count()),
                        help='Number of threads to use when processing audio files')
    args = parser.parse_args()

    args.input_dir = args.input_dir.rstrip('/')
    args.dest_dir = args.dest_dir.rstrip('/')

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
