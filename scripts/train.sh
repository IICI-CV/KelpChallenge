#!/bin/bash
# define the axes
dataset='kelp'
method='train_fabric'

HOME_N=/home/zb/WCS

data_root=$HOME_N/codes/SCANet/datasets/datasets/WHU

decoder_name='unet'
encoder_name='resnet18'

notes='bce_adamw_320x'

now=$(date +"%Y%m%d_%H%M%S")

exp=${decoder_name}'_'${encoder_name}'_'${notes}

config=configs/$dataset.yaml
train_id_path=$HOME_N/codes/Kelp/datasets/datasets/train_ids.txt
test_id_path=$HOME_N/codes/Kelp/datasets/datasets/val_ids.txt
val_id_path=$HOME_N/codes/Kelp/datasets/datasets/val_ids.txt
save_path=exp/$dataset/$method/$exp

mkdir -p $save_path

cp ${method}.py $save_path
cp datasets/${dataset}.py $save_path

python ${method}.py --exp_name=$exp \
    --config=$config \
    --dataset=$dataset \
    --train-id-path $train_id_path \
    --test-id-path $test_id_path \
    --val_id_path $val_id_path \
    --warmup_step 0 \
    --save-path $save_path \
    --encoder_name $encoder_name \
    --decoder_name $decoder_name \
    $(if [ -n "$exchanged" ]; then echo "--data_root=$data_root"; fi) \
    --port $2 2>&1 | tee $save_path/$now.log
# --train_from_best_submitted \
#         --best_submitted_path exp/a100micca3d/unimatch/unet_tiny_tiny_timm-resnest14d-sca__ce_999_adamw_lr9_${training_axis}/best.pth \