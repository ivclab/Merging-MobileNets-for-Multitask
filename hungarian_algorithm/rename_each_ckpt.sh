for END_LAYER in $(seq 13 13)
do
    cd ~/Documents/mobilenet_project/utils
    python rename_variables.py \
    --checkpoint_dir=/home/iis/Documents/mobilenet_project/merged_ckpt/${END_LAYER}/model_A \
    --replace_from='MobilenetV1' \
    --replace_to='MobilenetV1_M' \
    --out_dir_name=/home/iis/Documents/mobilenet_project/merged_ckpt/${END_LAYER}/MobilenetV1_M/MobilenetV1_M.ckpt
    python rename_variables.py \
    --checkpoint_dir=/home/iis/Documents/mobilenet_project/merged_ckpt/${END_LAYER}/model_B \
    --replace_from='MobilenetV1' \
    --replace_to='MobilenetV1_M' \
    --out_dir_name=/home/iis/Documents/mobilenet_project/merged_ckpt/${END_LAYER}/MobilenetV1_M_1/MobilenetV1_M_1.ckpt
    python rename_variables.py \
    --checkpoint_dir=/home/iis/Documents/mobilenet_project/merged_ckpt/${END_LAYER}/MobilenetV1_M_1 \
    --replace_from='Conv2d' \
    --replace_to='Conv2d_1' \
    --out_dir_name=/home/iis/Documents/mobilenet_project/merged_ckpt/${END_LAYER}/MobilenetV1_M_1/MobilenetV1_M_1.ckpt
    for LAYER in $(seq 1 ${END_LAYER})
    do
        python rename_variables.py \
        --checkpoint_dir=/home/iis/Documents/mobilenet_project/merged_ckpt/${END_LAYER}/MobilenetV1_M_1 \
        --replace_from=Conv2d_1_${LAYER}_pointwise \
        --replace_to=Conv2d_${LAYER}_pointwise \
        --out_dir_name=/home/iis/Documents/mobilenet_project/merged_ckpt/${END_LAYER}/MobilenetV1_M_1/MobilenetV1_M_1.ckpt
        python rename_variables.py \
        --checkpoint_dir=/home/iis/Documents/mobilenet_project/merged_ckpt/${END_LAYER}/MobilenetV1_M_1 \
        --replace_from=Conv2d_1_${LAYER}_depthwise \
        --replace_to=Conv2d_${LAYER}_depthwise_1 \
        --out_dir_name=/home/iis/Documents/mobilenet_project/merged_ckpt/${END_LAYER}/MobilenetV1_M_1/MobilenetV1_M_1.ckpt
    done
    cd ~/Documents/mobilenet_project/read_official_weight_ckpt
    python read_ckpt.py --end_layer=${END_LAYER}
done