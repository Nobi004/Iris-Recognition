from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import math

dense_blocks_num = 6
K = 12
L = 40
layers = 4

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON =1e-5

'''
(layers * 2) * dense_blocks_num + (dense_blocks_num - 1) + 1(init) + 1(fc)  
= (6 * 2) * 4 + (3 - 1) + 1 + 1
= 52
= L
'''

