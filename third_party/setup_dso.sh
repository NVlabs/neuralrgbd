'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''
#!/bin/bash

# clone DSO and apply patch 
git clone https://github.com/JakobEngel/dso 
cd dso
git reset --hard ae1d0b3d972367b0a384e587852bedcf5f414e69
cd .. 
patch -p1 < nnDepthDSO.patch 
mkdir dso/build && cd dso/build && cmake .. && make -j4
