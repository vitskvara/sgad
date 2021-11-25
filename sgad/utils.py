import os, sys
import torch.nn.functional as F
from torchvision.utils import save_image

def datadir(*args):
	file_path = os.path.realpath(__file__)
	return os.path.join(os.path.abspath(os.path.join(file_path, "../../../data")), *args)

def save_resize_img(x, path, n_row, sz=64):
    x = F.interpolate(x, (sz, sz))
    save_image(x.data, path, nrow=n_row, normalize=True, padding=2)
