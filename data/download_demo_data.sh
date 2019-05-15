#'''
#Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#Licensed under the CC BY-NC-SA 4.0 license
#
#author: Chao Liu <chaoliu1@cs.cmu.edu>
#'''

#!/bin/sh
echo "downloading demo data ..."
wget --quiet https://www.dropbox.com/s/occlpudcey2bb0e/scene0534_00.tar.gz?dl=0 -O scene0534_00.tar.gz 

echo "extracting..."
tar -xvf ./scene0534_00.tar.gz
