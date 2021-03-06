import numpy as np
from PIL import Image, ImageColor
from pathlib import Path

import torch
import torch.nn.functional as F

from torch import tensor
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset

import os
from sgad.utils import datadir
from sgad.utils import train_val_test_inds, split_data_labels, load_cifar10, load_wildlife_mnist, load_svhn2

class CIFAR10(Dataset):
    def __init__(self):
        raw_data = load_cifar10()
        
        self.data = tensor(raw_data[0]).float()
        self.labels = tensor(raw_data[1])
        self.T = transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                )

    def __getitem__(self, idx):
        return self.T(self.data[idx]), self.labels[idx]

    def __len__(self):
        return len(self.labels)

class WildlifeMNISTFixed(Dataset):
    def __init__(self, train=True):
        raw_data = load_wildlife_mnist(train=train)
        
        self.data = tensor(raw_data[0]).float()
        self.labels = tensor(raw_data[1])
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

class SVHN2(Dataset):
    def __init__(self):
        raw_data = load_svhn2()
        
        self.data = tensor(raw_data[0]).float()
        self.labels = tensor(raw_data[1])
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

def split_dataset(dataset, ratios=(0.6,0.2,0.2), seed=None, target_class=None, normalize=False):
    if target_class is None:
        split_inds = train_val_test_inds(np.arange(len(dataset)), ratios=ratios, seed=seed)
        (tr_data, tr_labels), (val_data, val_labels), (tst_data, tst_labels) = split_data_labels(dataset.data, dataset.labels, split_inds)
    else:
        # split normal data
        n_data = dataset.data[dataset.labels == target_class]
        n_labels = torch.zeros(n_data.shape[0]).long()
        n_split_inds = train_val_test_inds(np.arange(n_data.shape[0]), ratios=ratios, seed=seed)
        (n_tr_data, n_tr_labels), (n_val_data, n_val_labels), (n_tst_data, n_tst_labels) = split_data_labels(n_data, n_labels, n_split_inds)

        # split anomalous data
        a_data = dataset.data[dataset.labels != target_class]
        a_labels = torch.ones(a_data.shape[0]).long()
        a_split_inds = train_val_test_inds(np.arange(a_data.shape[0]), ratios=(0,0.5,0.5), seed=seed)
        (a_tr_data, a_tr_labels), (a_val_data, a_val_labels), (a_tst_data, a_tst_labels) = split_data_labels(a_data, a_labels, a_split_inds)

        # put it together
        tr_data = n_tr_data
        tr_labels = n_tr_labels
        val_data = torch.cat((n_val_data, a_val_data))
        val_labels = torch.cat((n_val_labels, a_val_labels))
        tst_data = torch.cat((n_tst_data, a_tst_data))
        tst_labels = torch.cat((n_tst_labels, a_tst_labels))

    return (Subset(tr_data, tr_labels, normalize=normalize), 
        Subset(val_data, val_labels, normalize=normalize), 
        Subset(tst_data, tst_labels, normalize=normalize)
        )

class Subset(Dataset):
    def __init__(self, data, labels, normalize=False):
        self.ims = data
        self.labels = labels
        self.normalize = normalize
        if normalize:
            self.T = transforms.Normalize(0.5, 0.5)
        else:
            self.T = lambda x: x

    def __getitem__(self, idx):
        ret = {
            'ims': self.T(self.ims[idx]),
            'labels': self.labels[idx],
        }

        return ret

    def __len__(self):
        return self.labels.shape[0]

def get_own_dataloaders(dataset_name, batch_size, workers, shuffle=True, train=True, **kwargs):
    if dataset_name == "cifar10":
        dataset = CIFAR10()
    elif dataset_name == "wildlife_mnist":
        dataset = WildlifeMNISTFixed(train=train)
    elif dataset_name == "svhn2":
        dataset = SVHN2()
    else:
        raise ValueError("f'{dataset_name} not known")

    tr_set, val_set, tst_set = split_dataset(dataset, **kwargs)
    tr_loader = DataLoader(tr_set, batch_size=batch_size,
                          shuffle=shuffle, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                          shuffle=False, num_workers=workers)
    tst_loader = DataLoader(tst_set, batch_size=batch_size,
                          shuffle=False, num_workers=workers)
    return tr_loader, val_loader, tst_loader

class ColoredMNIST(Dataset):
    def __init__(self, train, color_var=0.02):
        # get the colored mnist
        self.data_path = datadir('cgn_data/colored_mnist/mnist_10color_jitter_var_%.03f.npy'%color_var)
        data_dic = np.load(self.data_path, encoding='latin1', allow_pickle=True).item()

        if train:
            self.ims = data_dic['train_image']
            self.labels = tensor(data_dic['train_label'], dtype=torch.long)
        else:
            self.ims = data_dic['test_image']
            self.labels = tensor(data_dic['test_label'], dtype=torch.long)

        self.T = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32), Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ])

    def __getitem__(self, idx):
        ims, labels = self.T(self.ims[idx]), self.labels[idx]

        ret = {
            'ims': ims,
            'labels': labels,
        }

        return ret

    def __len__(self):
        return self.ims.shape[0]

class DoubleColoredMNIST(Dataset):

    def __init__(self, train=True):
        self.train = train
        self.mnist_sz = 32

        # get mnist
        mnist = datasets.MNIST(datadir("cgn_data/"), train=True, download=True)
        if train:
            ims, labels = mnist.data[:50000], mnist.targets[:50000]
        else:
            ims, labels = mnist.data[50000:], mnist.targets[50000:]

        self.ims_digit = torch.stack([ims, ims, ims], dim=1)
        self.labels = labels

        # colors generated by https://mokole.com/palette.html
        colors1 = [
            'darkgreen', 'darkblue', '#b03060',
            'orangered', 'yellow', 'burlywood', 'lime',
            'aqua', 'fuchsia', '#6495ed',
        ]
        # shift colors by X
        colors2 = [colors1[i-6] for i in range(len(colors1))]

        def get_rgb(x):
            t = torch.tensor(ImageColor.getcolor(x, "RGB"))/255.
            return t.view(-1, 1, 1)

        self.background_colors = list(map(get_rgb, colors1))
        self.object_colors = list(map(get_rgb, colors2))

        self.T = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        i = self.labels[idx] if self.train else np.random.randint(10)
        back_color = self.background_colors[i].clone()
        back_color += torch.normal(0, 0.01, (3, 1, 1))

        i = self.labels[idx] if self.train else np.random.randint(10)
        obj_color = self.object_colors[i].clone()
        obj_color += torch.normal(0, 0.01, (3, 1, 1))

        # get digit
        im_digit = (self.ims_digit[idx]/255.).to(torch.float32)
        im_digit = F.interpolate(im_digit[None,:], (self.mnist_sz, self.mnist_sz)).squeeze()
        im_digit = (im_digit > 0.1).to(int)  # binarize

        # plot digit onto the texture
        ims = im_digit*(obj_color) + (1 - im_digit)*back_color

        ret = {
            'ims': self.T(ims),
            'labels': self.labels[idx],
        }
        return ret

    def __len__(self):
        return self.labels.shape[0]

class WildlifeMNIST(Dataset):
    def __init__(self, train=True):
        self.train = train
        self.mnist_sz = 32
        inter_sz = 150

        # get mnist
        mnist = datasets.MNIST(datadir("cgn_data/"), train=True, download=True)
        if train:
            ims, labels = mnist.data[:50000], mnist.targets[:50000]
        else:
            ims, labels = mnist.data[50000:], mnist.targets[50000:]

        self.ims_digit = torch.stack([ims, ims, ims], dim=1)
        self.labels = labels

        # texture paths
        background_dir = Path(datadir('cgn_data/textures/background'))
        self.background_textures = sorted([im for im in background_dir.glob('*.jpg')])
        object_dir = Path(datadir('cgn_data/textures/object'))
        self.object_textures = sorted([im for im in object_dir.glob('*.jpg')])

        self.T_texture = transforms.Compose([
            transforms.Resize((inter_sz, inter_sz), Image.NEAREST),
            transforms.RandomCrop(self.mnist_sz, padding=3, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        labels_all = [self.labels[idx]]

        # get textures
        i = self.labels[idx] if self.train else np.random.randint(10)
        back_text = Image.open(self.background_textures[i])
        back_text = self.T_texture(back_text)
        labels_all.append(i)

        i = self.labels[idx] if self.train else np.random.randint(10)
        obj_text = Image.open(self.object_textures[i])
        obj_text = self.T_texture(obj_text)
        labels_all.append(i)

        # get digit
        im_digit = (self.ims_digit[idx]/255.).to(torch.float32)
        im_digit = F.interpolate(im_digit[None, :], (self.mnist_sz, self.mnist_sz)).squeeze()
        im_digit = (im_digit > 0.1).to(int)  # binarize

        # plot digit onto the texture
        ims = im_digit*(obj_text) + (1 - im_digit)*back_text

        ret = {
            'ims': ims,
            'labels': self.labels[idx],
            'labels_all': tensor(labels_all)
        }
        return ret

    def __len__(self):
        return self.labels.shape[0]

def get_dataloaders(dataset, batch_size, workers):
    if dataset == 'colored_MNIST':
        MNIST = ColoredMNIST
    elif dataset == 'double_colored_MNIST':
        MNIST = DoubleColoredMNIST
    elif dataset == 'wildlife_MNIST':
        MNIST = WildlifeMNIST
    else:
        raise TypeError(f"Unknown dataset: {dataset}")

    ds_train = MNIST(train=True)
    ds_test = MNIST(train=False)

    dl_train = DataLoader(ds_train, batch_size=batch_size,
                          shuffle=True, num_workers=workers)
    dl_test = DataLoader(ds_test, batch_size=batch_size*2,
                         shuffle=False, num_workers=workers)

    return dl_train, dl_test

TENSOR_DATASETS = ['colored_MNIST', 'colored_MNIST_counterfactual',
                   'double_colored_MNIST', 'double_colored_MNIST_counterfactual',
                   'wildlife_MNIST', 'wildlife_MNIST_counterfactual']

def get_tensor_dataloaders(dataset, batch_size=64):
    assert dataset in TENSOR_DATASETS, f"Unknown datasets {dataset}"

    if 'counterfactual' in dataset:
        tensor = torch.load(datadir(f'cgn_data/{dataset}.pth'))
        ds_train = TensorDataset(*tensor[:2])
        dataset = dataset.replace('_counterfactual', '')
    else:
        ds_train = TensorDataset(*torch.load(datadir(f'cgn_data/{dataset}_train.pth')))
    ds_test = TensorDataset(*torch.load(datadir(f'cgn_data/{dataset}_test.pth')))

    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=4,
                          shuffle=True, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size*10, num_workers=4,
                         shuffle=False, pin_memory=True)

    return dl_train, dl_test
