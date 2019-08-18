#'''
#Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#Licensed under the CC BY-NC-SA 4.0 license
#
#author: Chao Liu <chaoliu1@cs.cmu.edu>
#'''

#!/bin/bash 

#-- Test KV net on 7Scenes/ScanNet/KITTI using given pose --#
# parameters:
# - exp_name : the name for the experiment. The results will be saved in ../results/${exp_name}
# - sigma_soft_max: the sigma value used for getting the DPV. Should be the same as during training. 
# - t_win : the time window radius. The time window will be centered around the current frame
# - d_min, d_max : the minimal and maximal depth values
# - feature_dim: PSM feature dimension in DNet, should be the same as in the training session
# - dataset: the dataset name. Should one of scanNet, 7scenes, kitti
# - dataset_path: the path to the specified dataset
# - split_file: the spilt txt file, specifying which scenes/videos to use. For 7scenes, this is not needed (we will test on all videos)
# - model_path: the path to the trained model

# In this example, we test the trained model on one trajectory in the scanNet dataset
# The results will be saved in ../results/te/
CUDA_VISIBLE_DEVICES=0 python3 test_KVNet.py \
 		--exp_name te/ \
 		--sigma_soft_max 10\
 		--t_win 2 \
 		--d_min .1 \
 		--d_max 5 \
 		--feature_dim 64 \
 		--ndepth 64 \
 		--dataset scanNet \
        --dataset_path /datasets/scan-net-5-frame \
        --split_file ./mdataloader/scanNet_split/single.txt \
 		--model_path ./saved_models/kvnet_scannet.tar

# In this example, we test the trained model on trajectories in the KITTI dataset
# The results will be saved in ../results/te/
# CUDA_VISIBLE_DEVICES=0 python3 test_KVNet.py \
#  		--exp_name te/ \
#  		--sigma_soft_max 10\
# 		--t_win 2 \
# 		--d_min 1 \
# 		--d_max 60 \
#  		--feature_dim 64 \
#  		--ndepth 64 \
# 		--dataset kitti \
# 		--dataset_path /datasets/kitti \
#         --split_file ./mdataloader/kitti_split/testing.txt \
#  		--model_path ./saved_models/kvnet_kitti.tar




