##### Parameters for testing 

The parameters for `test_KVNet.py`:

`exp_name`: The name for the experiment. The results will be saved in ../results/`${exp_name}`

`sigma_soft_max`: the sigma value used for getting the DPV. Should be the same as during training. 

`t_win`: the time window radius. The time window will be centered around the current frame

`d_min, d_max` : the minimal and maximal depth values

`feature_dim`: PSM feature dimension in DNet, should be the same as in the training session

`dataset`: the dataset name. Should one of `scanNet, 7scenes, kitti`

`dataset_path`: the path to the specified dataset

`split_file`: the spilt txt file, specifying which scenes/videos to use. 

`model_path`: the path to the trained model


##### Test on ScanNet
Suppose the decoded ScanNet dataset, with 5-frame interval, is at `/datasets/scan-net-5-frame`, to test on ScanNet dataset:
```
CUDA_VISIBLE_DEVICES=0 python3 test_KVNet.py \
 		--exp_name te_scannet/ \
 		--sigma_soft_max 10\
 		--t_win 2 \
 		--d_min .1 \
 		--d_max 5 \
 		--feature_dim 64 \
 		--ndepth 64 \
 		--dataset scanNet \
        --dataset_path /datasets/scan-net-5-frame \
        --split_file ./mdataloader/scanNet_split/scannet_val.txt \
 		--model_path ./saved_models/kvnet_scannet.tar
```
The results will be saved at `../results/te_scannet`.

##### Test on KITTI
Suppose the KITTI dataset is at `/datasets/kitti`, and the folders are organized as
```
/datasets/kitti/rawdata
/datasets/kitti/train
/datasets/kitti/val
```
to test on KITTI dataset:
```
CUDA_VISIBLE_DEVICES=0 python3 test_KVNet.py \
 		--exp_name te_kitti/ \
 		--sigma_soft_max 10\
 		--t_win 2 \
 		--d_min 1 \
 		--d_max 60 \
 		--feature_dim 64 \
 		--ndepth 64 \
 		--dataset kitti \
        --dataset_path /datasets/kitti \
        --split_file ./mdataloader/kitti_split/test_eigen.txt \
 		--model_path ./saved_models/kvnet_kitti.tar
```
The results will be saved at `../results/te_kitti`.

##### Test on 7Scenes 
Suppose the 7Scenes dataset is at `/datasets/7scenes`,
```
CUDA_VISIBLE_DEVICES=0 python3 test_KVNet.py \
 		--exp_name te_7scenes/ \
 		--sigma_soft_max 10\
 		--t_win 2 \
 		--d_min .1 \
 		--d_max 5 \
 		--feature_dim 64 \
 		--ndepth 64 \
 		--dataset 7scenes \
        --dataset_path /datasets/7scenes \
 		--model_path ./saved_models/kvnet_scannet.tar
```
The results will be saved at `../results/te_7scenes`.
