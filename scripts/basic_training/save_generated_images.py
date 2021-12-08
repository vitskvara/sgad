# generic libraries
import numpy as np
import torch
import os, sys
import argparse
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('input',
                    help='name of the input file')
args = parser.parse_args()

# sgad
SGADHOME='/home/skvara/work/counterfactual_ad/sgad'
sys.path.append(SGADHOME)
from sgad.utils import datadir
from sgad.utils import save_resize_img

# load the generated data
dataf = args.input
counterfactuals = torch.load(dataf)

# save images
modelid = os.path.basename(dataf).split("_")[0]
outf = os.path.join(os.path.dirname(dataf), modelid+"_counterfactual.png")
save_resize_img(counterfactuals[0][0:100,:,:,:].data, outf, n_row=10)
