##### Customize training

The parameters for `train_KVNet.py`:
`exp_name`: The name for the experiment. The trained model will be saved in `saved_model/{exp_name}` by default

`nepoch`: number of epochs.

`LR`: learning rate

`grad_clip, grad_clip_max`: if perform the gradient clipping and its threshold.

`RNet`: if train the refinement network

`batch_size`: The batch size for training.  How many videos we will feed into the memory at one time. Now we only support feeding one video per GPU (taking ~6 GB of memory for training). default=0, it means we will use all available GPUs.

`sigma_soft_max`: the sigma value used for getting the DPV. 

`t_win`: the time window radius. The time window will be centered around the current frame

`d_min, d_max` : the minimal and maximal depth values

`feature_dim`: PSM feature dimension in DNet

`dataset`: the dataset name. Should one of `scanNet, 7scenes, kitti`. 

`dataset_path`: the path to the specified dataset

`model_path`: the path to the trained model

So suppose the 5-frame-interval scanNet dataset is stored at `/datasets/scan-net-5-frame`, to train the model used in the paper:
```
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
```


