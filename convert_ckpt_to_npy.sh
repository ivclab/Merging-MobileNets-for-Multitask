#!/bin/bash

python ./hungarian_algorithm/convert_ckpt_to_npy.py \
    --model_meta='./checkpoints/cubs_cropped/model.ckpt-30000.meta' \
    --model_ckpt='./checkpoints/cubs_cropped' \
    --npy_our_dir='./hungarian_algorithm/origin_npy/cubs_cropped/cubs_cropped.npy'

python ./hungarian_algorithm/convert_ckpt_to_npy.py \
    --model_meta='./checkpoints/flowers_102/model.ckpt-30000.meta' \
    --model_ckpt='./checkpoints/flowers_102' \
    --npy_our_dir='./hungarian_algorithm/origin_npy/flowers_102/flowers_102.npy'

