#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 17:24:37 2021

@author: tan
"""

import numpy as np
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import torch



DATA_PATH = 'data/modelnet40_normal_resampled/'

TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='train',
                                                 normal_channel=False)
TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='test',
                                                normal_channel=False)

num_train_data = TRAIN_DATASET.__len__()
train_final = []
for i in range(num_train_data):
    print("Aggregating number ", i, " instance...")
    cur_train_data = TRAIN_DATASET._get_item(i)[0]
    train_final.append(cur_train_data)

train_final = np.asarray(train_final)
np.save('data/train.npy',train_final)

num_test_data = TEST_DATASET.__len__()
test_final = []
for i in range(num_test_data):
    print("Aggregating number ", i, " instance...")
    cur_test_data = TEST_DATASET._get_item(i)[0]
    test_final.append(cur_test_data)

test_final = np.asarray(test_final)
np.save('data/test.npy',test_final)