#!/bin/bash

for folder_index in D20 D21 D22 D23 D24 D25 D26 D27
do 
    python ./local/convert_kconfspeech.py \
        --input_dir ./data/train/KconfSpeech/${folder_index} \
        --dest_dir ./data/KconfSpeech/${folder_index}-wav \
        --output_json ./data/KconfSpeech/${folder_index}-wav.json \
        --target_sr 16000 \
        --speed 0.9 1.1 \
	--overwrite
done

python ./local/convert_kconfspeech.py \
    --input_dir ./data/valid/ \
    --dest_dir ./data/KconfSpeech/valid-wav \
    --output_json ./data/KconfSpeech/valid-wav.json \
    --target_sr 16000 \
    --overwrite

python ./local/convert_kconfspeech.py \
    --input_dir ./data/test/ \
    --dest_dir ./data/KconfSpeech/test-wav \
    --output_json ./data/KconfSpeech/test-wav.json \
    --target_sr 16000 \
    --overwrite