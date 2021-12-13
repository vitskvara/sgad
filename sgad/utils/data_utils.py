import os, sys
import pickle
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image

def datadir(*args):
    file_path = os.path.realpath(__file__)
    return os.path.join(os.path.abspath(os.path.join(file_path, "../../../../data")), *args)

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

def load_wildlife_mnist(train=True):
    path = datadir("raw_datasets/wildlife_MNIST")
    if train:
        data = np.load(os.path.join(path, "data.npy"))
        labels = np.load(os.path.join(path, "labels.npy"))
    else:
        data = np.load(os.path.join(path, "data_test.npy"))
        labels = np.load(os.path.join(path, "labels_test.npy"))
    return data, labels

def train_val_test_inds(indices, ratios=(0.6,0.2,0.2), seed=None):
    if (sum(ratios) != 1.0 or len(ratios) != 3):
        raise ValueError("ratios must be a vector of length 3 that sums up to 1")

    # set seed
    rng = np.random.RandomState() if seed == None else np.random.RandomState(seed)
    _indices = rng.permutation(indices)

    # set number of samples in individual subsets
    n = len(indices)
    ns = np.cumsum(np.floor(n*np.array(ratios)).astype(int))

    # return the sets of indices
    return _indices[0:ns[0]], _indices[ns[0]+1:ns[1]], _indices[ns[1]+1:ns[2]]

def split_data_labels(data, labels, split_inds):
    tr_inds, val_inds, tst_inds = split_inds
    tr_data, val_data, tst_data = data[tr_inds], data[val_inds], data[tst_inds]
    tr_labels, val_labels, tst_labels = labels[tr_inds], labels[val_inds], labels[tst_inds]

    return (tr_data, tr_labels), (val_data, val_labels), (tst_data, tst_labels)
