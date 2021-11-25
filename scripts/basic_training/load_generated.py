import numpy as np
import torch
import os, sys
CGNHOME='/home/skvara/work/counterfactual_ad/counterfactual_generative_networks'
sys.path.append(CGNHOME)

f = os.path.join(CGNHOME, "mnists/data/wildlife_MNIST_counterfactual.pth")
counterfactuals = torch.load(f)

from torchvision.utils import save_image

outp = "/home/skvara/work/counterfactual_ad/data/cgn_generated/wildlife_MNIST"
outf = os.path.join(outp, "test.png")
save_image(counterfactuals[0][0:100,:,:,:].data, outf, n_row=3)

def save(x, path, n_row, sz=64):
    x = F.interpolate(x, (sz, sz))
    save_image(x.data, path, nrow=n_row, normalize=True, padding=2)
