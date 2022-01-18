import os, sys

# sgad
SGADHOME='/home/skvara/work/counterfactual_ad/sgad'
sys.path.append(SGADHOME)

import sgad
from sgad.utils import train_val_test_inds, load_cifar10, split_data_labels
from sgad.cgn import CIFAR10, SVHN2
from torch import tensor
from torch.utils.data import DataLoader

import numpy as np

indices = np.array(range(500))

tr_inds, val_inds, tst_inds = train_val_test_inds(indices)

inds1 = train_val_test_inds(indices, seed=3)
inds2 = train_val_test_inds(indices, seed=3)

cifar = CIFAR10()
tr_loader, val_loader, tst_loader = sgad.cgn.dataloader.split_dataset(cifar, seed=2)

loader = DataLoader(tr_loader, batch_size=13, shuffle=True, num_workers=4, pin_memory=True)

batch = next(iter(loader))
batch["ims"]

from sgad.cgn.dataloader import get_dataloaders

dl,_ = get_dataloaders('wildlife_MNIST', 32, 12)

bbatch = next(iter(dl))

svhn2 = SVHN2()
tr_set, val_set, tst_set = sgad.cgn.dataloader.split_dataset(svhn2, seed=2, target_class=0)

batch_size = 128
shuffle = True
workers = 12
loader = DataLoader(tr_set, batch_size=batch_size,
                          shuffle=shuffle, num_workers=workers)

batch = next(iter(loader))

sgad.utils.save_resize_img(batch['ims'], "test.png", 8)
