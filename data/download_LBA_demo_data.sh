#'''
#Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#Licensed under the CC BY-NC-SA 4.0 license
#
#author: Chao Liu <chaoliu1@cs.cmu.edu>
#'''

#!/bin/bash
echo "downloading demo data ..."
wget --quiet -O 7scenes_office_seq_01.tar.gz https://www.dropbox.com/s/hrj09azsgomyqfx/office_seq_01.tar.gz?dl=0

echo "extracting..."
tar -xvf 7scenes_office_seq_01.tar.gz
