#!/bin/bash

for folder_index in D50 D51 D52
do 
    python ./local/convert_krespspeech.py \
        --input_dir ./data/train/KresSpeech/${folder_index} \
        --dest_dir ./data/KrespSpeech/${folder_index}-wav \
        --output_json ./data/KrespSpeech/${folder_index}-wav.json \
        --target_sr 16000 \
        --speed 0.9 1.1 \
	--overwrite
done

python ./local/convert_krespspeech.py \
    --input_dir ./data/valid/ \
    --dest_dir ./data/KrespSpeech/valid-wav \
    --output_json ./data/KrespSpeech/valid-wav.json \
    --target_sr 16000 \
    --overwrite

python ./local/convert_krespspeech.py \
    --input_dir ./data/test/ \
    --dest_dir ./data/KrespSpeech/test-wav \
    --output_json ./data/KrespSpeech/test-wav.json \
    --target_sr 16000 \
    --overwrite