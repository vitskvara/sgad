import numpy as np
import torch
import os, sys
SGADHOME='/home/skvara/work/counterfactual_ad/sgad'
sys.path.append(SGADHOME)
from sgad.utils import datadir
from sgad.cgn.dataloader import get_dataloaders
from torchvision import datasets

# setup save path
outdir = datadir("raw_datasets/wildlife_MNIST")
os.makedirs(outdir, exist_ok=True)

# get the dataloaders
dl_train, dl_test = get_dataloaders("wildlife_MNIST", 500, 12)
dl_test.dataset.train = True

# now get a dataset of 60000 samples
x = []
y = []
for batch in dl_train:
    x.append(batch['ims'])
    y.append(batch['labels'])
for batch in dl_test:
    x.append(batch['ims'])
    y.append(batch['labels'])
x = torch.cat(x)
y = torch.cat(y)

# now save the np array
np.save(os.path.join(outdir, "data.npy"), np.array(x))
np.save(os.path.join(outdir, "labels.npy"), np.array(y))

# also generate test dataset
dl_train.dataset.train = False
dl_test.dataset.train = False
x = []
y = []

for batch in dl_train:
    x.append(batch['ims'])
    y.append(batch['labels_all'])

for batch in dl_test:
    x.append(batch['ims'])
    y.append(batch['labels_all'])

x = torch.cat(x)
y = torch.cat(y)

# now save the np array
np.save(os.path.join(outdir, "data_test.npy"), np.array(x))
np.save(os.path.join(outdir, "labels_test.npy"), np.array(y))
