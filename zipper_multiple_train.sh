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


for i in $(seq 1 13); do
  echo "Mutiple Task Train Layer:"$i

  python mutiple_task_train.py \
    --dataset_dir_A=./datasets/cubs_cropped \
    --dataset_dir_B=./datasets/flowers_102 \
    --train_dir=./checkpoints/multiple_cubs_cropped_flowers_102/$i \
    --dataset_name_A=cubs_cropped \
    --dataset_name_B=flowers_102 \
    --dataset_split_name_A=train \
    --dataset_split_name_B=train \
    --number_of_steps=4000 \
    --gpu_memory_fraction=0.7 \
    --model_name=mobilenet_v1 \
    --merged_model_scope=MobilenetV1_M \
    --model_scope_A_teacher=MobilenetV1_cubs_cropped \
    --model_scope_B_teacher=MobilenetV1_flowers_102 \
    --checkpoint_model_scope_A=MobilenetV1_cubs_cropped \
    --checkpoint_model_scope_B=MobilenetV1_flowers_102 \
    --checkpoint_path_A=./hungarian_algorithm/merged_ckpt/cubs_cropped_and_flowers_102/cubs_cropped/$i \
    --checkpoint_path_B=./hungarian_algorithm/merged_ckpt/cubs_cropped_and_flowers_102/flowers_102/$i \
    --checkpoint_path_A_teacher=./checkpoints/cubs_cropped \
    --checkpoint_path_B_teacher=./checkpoints/flowers_102 \
    --preprocessing_name=inception_preprocessing \
    --conv2d_0_scope_A=Conv2d_0_cubs_cropped \
    --conv2d_0_scope_B=Conv2d_0_flowers_102 \
    --depthwise_scope_A=_depthwise_cubs_cropped \
    --depthwise_scope_B=_depthwise_flowers_102 \
    --pointwise_scope_A=_pointwise_cubs_cropped \
    --pointwise_scope_B=_pointwise_flowers_102 \
    --logits_scope_A=Logits_cubs_cropped \
    --logits_scope_B=Logits_flowers_102 \
    --pointwise_merged_mask=${MASK_LIST[$i]} \
    --ignore_missing_vars=False




