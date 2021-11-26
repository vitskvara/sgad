import os, sys
import pickle
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image

def datadir(*args):
    file_path = os.path.realpath(__file__)
    return os.path.join(os.path.abspath(os.path.join(file_path, "../../../data")), *args)

def save_resize_img(x, path, n_row, sz=64):
    x = F.interpolate(x, (sz, sz))
    save_image(x.data, path, nrow=n_row, normalize=True, padding=2)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10():
    path = datadir("raw_datasets/cifar10/cifar-10-batches-py")
    fs = os.listdir(path)
    fs = filter(lambda x: x.find('_batch') != -1, fs)
    all_data = [y for y in map(lambda x: unpickle(os.path.join(path, x)), fs)]
    data = np.concatenate([x[b'data'].reshape(10000, 3, 32, 32) for x in all_data])/255
    labels = np.concatenate([x[b'labels'] for x in all_data])
    return data, labels
