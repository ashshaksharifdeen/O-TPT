#!/bin/bash

data_root='/home/ashashak/VLM-calibration/C-TPT/dataset'
testsets=$1
csv_loc='/home/ashashak/VLM-calibration/C-TPT/log/test_otpt_finegrained_tp-ip-disp.csv'
#arch= RN50
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a
run_type=tpt_otpt
lambda_term=18

python ./otpt_classification.py ${data_root} --test_sets ${testsets} --csv_log ${csv_loc} \
-a ${arch} -b ${bs} --gpu 2 \
--tpt --ctx_init ${ctx_init} --run_type ${run_type} --lambda_term ${lambda_term} \
