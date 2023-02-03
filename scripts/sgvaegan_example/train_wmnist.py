import sgad
import numpy as np
import os
import torch
from torch import nn
from pathlib import Path

from sgad.sgvae import SGVAEGAN
from sgad.utils import load_wildlife_mnist_split, datadir
from sgad.utils import save_cfg, load_cfg

#sgad.utils.save_resize_img(torch.tensor(data_anomalous[:5]),"test.png",1)
# 32131929, 35044565
from sgad.utils import train_val_test_inds, split_data_labels


# setup
normal_class = 0
seed = 4
n_epochs = 100

# setup path for the model to be saved
outpath = Path(datadir("test_models/"))
outpath.mkdir(parents=True, exist_ok=True)
nfiles = len(os.listdir(outpath))
outpath = os.path.join(outpath, f"run_{nfiles+1}")

# load the data - note that the images are in [-1,1] range
data = load_wildlife_mnist_split(normal_class, seed=seed, train=True, denormalize=False)
(tr_x, tr_y, tr_c), (val_x, val_y, val_c), (tst_x, tst_y, tst_c) = data

# use the defaults for training of wildlife mnist
model = SGVAEGAN(init_seed=35044565)
model.fit(tr_x, n_epochs=100, save_path=outpath)
