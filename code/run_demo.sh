#'''
#Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#Licensed under the CC BY-NC-SA 4.0 license
#
#author: Chao Liu <chaoliu1@cs.cmu.edu>
#'''

#!/bin/bash 
# run demo, suppose the downloaded demo data is in ../data

CUDA_VISIBLE_DEVICES=0 python3 test_KVNet.py \
 		--exp_name demo/ \
 		--sigma_soft_max 10\
 		--t_win 2 \
 		--d_min .1 \
 		--d_max 5 \
 		--feature_dim 64 \
 		--ndepth 64 \
 		--dataset scanNet \
        --dataset_path ../data/datasets/scan-net-5-frame \
 		--model_path ./saved_models/kvnet_scannet.tar \
        --split_file ./mdataloader/scanNet_split/single.txt
