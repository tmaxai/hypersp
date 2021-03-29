#!/bin/bash

for folder_index in D60 D61 D62
do 
    python ./local/convert_ktelspeech.py \
        --input_dir ./data/train/KtelSpeech/${folder_index} \
        --dest_dir ./data/KtelSpeech/${folder_index}-wav \
        --output_json ./data/KtelSpeech/${folder_index}-wav.json \
        --target_sr 16000 \
        --speed 0.9 1.1 \
	--overwrite
done

python ./local/convert_ktelspeech.py \
    --input_dir ./data/valid/ \
    --dest_dir ./data/KtelSpeech/valid-wav \
    --output_json ./data/KtelSpeech/valid-wav.json \
    --target_sr 16000 \
    --overwrite

python ./local/convert_ktelspeech.py \
    --input_dir ./data/test/ \
    --dest_dir ./data/KtelSpeech/test-wav \
    --output_json ./data/KtelSpeech/test-wav.json \
    --target_sr 16000 \
    --overwrite