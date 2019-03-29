#!/bin/bash

for LAYER in $(seq 1 13)
do  
    python ./hungarian_algorithm/merge_layer.py \
    --start_layer=0 \
    --end_layer=${LAYER} \
    --model_A='cubs_cropped' \
    --model_B='flowers_102' \
    --model_meta_A='./checkpoints/cubs_cropped/model.ckpt-30000.meta' \
    --model_ckpt_A='./checkpoints/cubs_cropped' \
    --model_meta_B='./checkpoints/flowers_102/model.ckpt-30000.meta' \
    --model_ckpt_B='./checkpoints/flowers_102' \
    --merged_npy_out_dir_A='./hungarian_algorithm/merged_npy/cubs_cropped_and_flowers_102/cubs_cropped/' \
    --merged_npy_out_dir_B='./hungarian_algorithm/merged_npy/cubs_cropped_and_flowers_102/flowers_102/'
done

for LAYER in $(seq 1 13)
do  
    python ./hungarian_algorithm/assign_ckpt.py \
    --model_A='cubs_cropped' \
    --model_B='flowers_102' \
    --model_meta_A='./checkpoints/cubs_cropped/model.ckpt-30000.meta' \
    --model_ckpt_A='./checkpoints/cubs_cropped' \
    --model_meta_B='./checkpoints/flowers_102/model.ckpt-30000.meta' \
    --model_ckpt_B='./checkpoints/flowers_102' \
    --merged_npy_A='./hungarian_algorithm/merged_npy/cubs_cropped_and_flowers_102/cubs_cropped/'${LAYER}'/merged_cubs_cropped.npy' \
    --merged_npy_B='./hungarian_algorithm/merged_npy/cubs_cropped_and_flowers_102/flowers_102/'${LAYER}'/merged_flowers_102.npy' \
    --merged_ckpt_out_dir_A='./hungarian_algorithm/merged_ckpt/cubs_cropped_and_flowers_102/cubs_cropped/'${LAYER}'/' \
    --merged_ckpt_out_dir_B='./hungarian_algorithm/merged_ckpt/cubs_cropped_and_flowers_102/flowers_102/'${LAYER}'/'
done
