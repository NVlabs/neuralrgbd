#'''
#Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#Licensed under the CC BY-NC-SA 4.0 license
#
#author: Chao Liu <chaoliu1@cs.cmu.edu>
#'''

#!/bin/bash

# train on scanNet #
python3 train_KVNet.py \
		--exp_name tr_scanNet/ \
		--nepoch 10 \
		--sigma_soft_max 10\
		--LR 1e-5 \
		--t_win 2 \
		--d_min .1 \
		--d_max 5. \
		--feature_dim 64 \
		--ndepth 64 \
		--grad_clip \
		--grad_clip_max 2. \
		--RNet \
		--batch_size 0 \
		--dataset scanNet \
        --dataset_path /datasets/scan-net-5-frame
