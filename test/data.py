import os, sys

# sgad
SGADHOME='/home/skvara/work/counterfactual_ad/sgad'
sys.path.append(SGADHOME)

import sgad
from sgad.utils import train_val_test_inds, load_cifar10, split_data_labels
from sgad.cgn import CIFAR10, CIFAR10Subset
from torch import tensor
from torch.utils.data import DataLoader

import numpy as np

indices = np.array(range(500))

tr_inds, val_inds, tst_inds = train_val_test_inds(indices)

inds1 = train_val_test_inds(indices, seed=3)
inds2 = train_val_test_inds(indices, seed=3)

cifar = CIFAR10()
tr_loader, val_loader, tst_loader = cifar.split(seed=2)

loader = DataLoader(tr_loader, batch_size=13, shuffle=True, num_workers=4, pin_memory=True)

batch = next(iter(loader))
batch["ims"]

from sgad.cgn.dataloader import get_dataloaders

dl,_ = get_dataloaders('wildlife_MNIST', 32, 12)

bbatch = next(iter(dl))
