#!/bin/bash

MAPPER_PATH='/public/home/qinxy/yimj/V2A/train/checkpoints_mapper/MLPModel5_final.pth'
VIDEO_PATH='/public/home/qinxy/AudioData/VGGsound'
OUTPUT_PATH='/public/home/qinxy/yimj/V2A_new/output/MLP1'

python pipeline.py \
    --mapper_path "$MAPPER_PATH" \
    --video_folder "$VIDEO_PATH" \
    --output "$OUTPUT_PATH" \
