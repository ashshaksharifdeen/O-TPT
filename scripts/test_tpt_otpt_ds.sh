#!/bin/bash

data_root='./dataset'
testsets=$1
csv_loc='./log/test_otpt_finegrained_image-disperssion-only.csv'
#arch=RN50
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a
run_type=tpt_otpt
lambda_term=2

python ./otpt_classification.py ${data_root} --test_sets ${testsets} --csv_log ${csv_loc} \
-a ${arch} -b ${bs} --gpu 2 \
--tpt --ctx_init ${ctx_init} --run_type ${run_type} --I_augmix --lambda_term ${lambda_term} \
