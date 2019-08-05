#!/bin/bash

MASK_LIST=(
    "0,0,0,0,0,0,0,0,0,0,0,0,0,0" 
    "0,1,0,0,0,0,0,0,0,0,0,0,0,0"
    "0,1,1,0,0,0,0,0,0,0,0,0,0,0"
    "0,1,1,1,0,0,0,0,0,0,0,0,0,0"
    "0,1,1,1,1,0,0,0,0,0,0,0,0,0"
    "0,1,1,1,1,1,0,0,0,0,0,0,0,0"
    "0,1,1,1,1,1,1,0,0,0,0,0,0,0"
    "0,1,1,1,1,1,1,1,0,0,0,0,0,0"
    "0,1,1,1,1,1,1,1,1,0,0,0,0,0"
    "0,1,1,1,1,1,1,1,1,1,0,0,0,0"
    "0,1,1,1,1,1,1,1,1,1,1,0,0,0"
    "0,1,1,1,1,1,1,1,1,1,1,1,0,0"
    "0,1,1,1,1,1,1,1,1,1,1,1,1,0"
    "0,1,1,1,1,1,1,1,1,1,1,1,1,1"
)


for i in $(seq 2 13); do
  echo "LAYER:"$i
  j=$((i+1))
  python mobilenet_v1_eval.py \
    --dataset_name=cubs_cropped \
    --dataset_split_name=test \
    --checkpoint_path=./checkpoints/multiple_cubs_cropped_flowers_102/$i \
    --dataset_dir=./datasets/cubs_cropped \
    --model_scope=MobilenetV1_M \
    --conv2d_0_scope=Conv2d_0_cubs_cropped \
    --depthwise_scope=_depthwise_cubs_cropped \
    --pointwise_scope=_pointwise_cubs_cropped \
    --logits_scope=Logits_cubs_cropped \
    --pointwise_merged_mask=${MASK_LIST[i]}

  python mobilenet_v1_eval.py \
    --dataset_name=flowers_102 \
    --dataset_split_name=test \
    --checkpoint_path=./checkpoints/multiple_cubs_cropped_flowers_102/$i \
    --dataset_dir=./datasets/flowers_102 \
    --model_scope=MobilenetV1_M \
    --conv2d_0_scope=Conv2d_0_flowers_102 \
    --depthwise_scope=_depthwise_flowers_102 \
    --pointwise_scope=_pointwise_flowers_102 \
    --logits_scope=Logits_flowers_102 \
    --pointwise_merged_mask=${MASK_LIST[i]}
done
