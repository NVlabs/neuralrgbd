'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

import PIL.Image as image
def export2pgm( fpath , im ):
    image.fromarray(im).convert('I').save(fpath)
