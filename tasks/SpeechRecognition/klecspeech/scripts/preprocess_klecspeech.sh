#!/bin/bash

for folder_index in D01 D02 D03 D04 D05 D06 D07 D08 D09 D10 D11 D12 D13 D14 D15 D16 D17 D18 D19 D99
do 
    python ./local/convert_klecspeech.py \
        --input_dir ./data/train/KlecSpeech/${folder_index} \
        --dest_dir ./data/KlecSpeech/${folder_index}-wav \
        --output_json ./data/KlecSpeech/${folder_index}-wav.json \
        --target_sr 16000 \
        --speed 0.9 1.1 \
	--overwrite
done

python ./local/convert_klecspeech.py \
    --input_dir ./data/valid/ \
    --dest_dir ./data/KlecSpeech/valid-wav \
    --output_json ./data/KlecSpeech/valid-wav.json \
    --target_sr 16000 \
    --overwrite

python ./local/convert_klecspeech.py \
    --input_dir ./data/test/ \
    --dest_dir ./data/KlecSpeech/test-wav \
    --output_json ./data/KlecSpeech/test-wav.json \
    --target_sr 16000 \
    --overwrite