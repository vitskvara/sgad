import os, sys

# sgad
SGADHOME='/home/skvara/work/counterfactual_ad/sgad'
sys.path.append(SGADHOME)

import sgad
from sgad.utils import train_val_test_inds, load_cifar10, split_data_labels
from sgad.cgn import CIFAR10, CIFAR10Subset

import numpy as np

indices = np.array(range(500))

tr_inds, val_inds, tst_inds = train_val_test_inds(indices)

inds1 = train_val_test_inds(indices, seed=3)
inds2 = train_val_test_inds(indices, seed=3)

cifar = CIFAR10()
tr_set, val_set, tst_set = cifar.split(seed=2)

