#!/bin/bash
if [ -z "$1" ]
then
    blr=2e-4
else
    blr=$1
fi

audioset_train_all_video_json=/workspace/AudioMAE-TE/datafiles/pretrain/train_all_video.json
audioset_train_all_json=/workspace/AudioMAE-TE/datafiles/taiwan_birds/pretrain/6_days_40_files/filtered/train_birds_project.json
#audioset_train_all_json=/workspace/AudioMAE-TE/datafiles/taiwan_birds/pretrain/100000_datas/train_birds_project.json
audioset_label=/workspace/AudioMAE-TE/datafiles/taiwan_birds/pretrain/6_days_40_files/class_labels_indices.csv


dataset=taiwan_birds

python3 submitit_pretrain.py \
--print_freq 100 \
--nodes 1 \
--ngpus 1 \
--batch_size 64 \
--norm_pix_loss True \
--model mae_vit_base_patch16 \
--mask_ratio 0.8 \
--epochs 1 \
--warmup_epochs 3 \
--save_every_epoch 10 \
--blr $blr --weight_decay 0.0001 \
--dataset $dataset \
--data_train $audioset_train_all_json \
--label_csv $audioset_label \
--roll_mag_aug True \
--distributed True \
--decoder_mode 1 \

#sleep 20
#nvidia-smi



