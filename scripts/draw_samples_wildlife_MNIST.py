import numpy as np
import os, sys
import torch
SGADHOME='/home/skvara/work/counterfactual_ad/sgad'
sys.path.append(SGADHOME)
from sgad.utils import datadir
from sgad.utils import save_resize_img

# train wildlife mnist
datapath = datadir("raw_datasets/wildlife_MNIST")
data = np.load(os.path.join(datapath, "data.npy"))
labels = np.load(os.path.join(datapath, "labels.npy"))

rng = np.random.default_rng()
x = []
for i in range(10):
    x.append(rng.choice(data[labels == i], 10))

x = np.concatenate(x)
save_resize_img(torch.tensor(x), os.path.join(datapath, "train.png"), 10)

# test wildlife mnist
datapath = datadir("raw_datasets/wildlife_MNIST")
data = np.load(os.path.join(datapath, "data_test.npy"))
labels = np.load(os.path.join(datapath, "labels_test.npy"))

rng = np.random.default_rng()
x = []
for i in range(10):
    x.append(rng.choice(data[np.array([l[0] for l in labels]) == i], 10))
x = np.concatenate(x)
save_resize_img(torch.tensor(x), os.path.join(datapath, "test_digit.png"), 10)

x = []
for i in range(10):
    x.append(rng.choice(data[np.array([l[1] for l in labels]) == i], 10))
x = np.concatenate(x)
save_resize_img(torch.tensor(x), os.path.join(datapath, "test_background.png"), 10)

x = []
for i in range(10):
    x.append(rng.choice(data[np.array([l[2] for l in labels]) == i], 10))
x = np.concatenate(x)
save_resize_img(torch.tensor(x), os.path.join(datapath, "test_texture.png"), 10)
