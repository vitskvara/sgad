import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
import warnings
import functools
import random

from sgad.cgn.models.cgn import Reshape, UpsampleBlock, lin_block, shape_layers, texture_layers
from sgad.cgn.models.cgn import get_norm_layer
from sgad.utils import Subset

def logreg_fit(X, y):
    """
        logreg_fit(X, y)

        X must have shape (nsamples, nregressors). Fits with zero intercept. Returns coeffs and the solver.
    """
    clf = LogisticRegression(fit_intercept=False, class_weight='balanced').fit(X, y)
    alpha = clf.coef_[0]
    return alpha.reshape(4), clf

def logreg_prob(X, alpha):
    """
        logreg_prob(X, alpha)

       X must have shape (nsamples, nregressors), alpha must have shape (nregressors,). 
    """
    ax = np.matmul(X, alpha)
    with warnings.catch_warnings(): # to filter out the ugly exp overflow error
        warnings.simplefilter("ignore")
        return 1/(1+np.exp(ax))

class Mean(nn.Module):
    def __init__(self, *args):
        super(Mean, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.mean(self.shape) 

def ConvBlock(in_channels, out_channels, activation="leakyrelu", bn=True, bias=False, stride=2):
    if activation == "leakyrelu":
        af = nn.LeakyReLU(0.2)
    elif activation == "tanh":
        af = nn.Tanh()
    else:
        raise ValueError(f"unimplemented activation function {activation}")
    res =  [nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)]
    res.append(nn.BatchNorm2d(out_channels)) if bn else None
    res.append(af)
    return res

def Encoder(z_dim, img_channels, h_channels, img_dim, activation="leakyrelu", batch_norm=True, n_layers=3):
    out_dim = img_dim // 8
    lin_dim = h_channels*4*out_dim*out_dim
    # correct activation
    if activation == "leakyrelu":
        af = nn.LeakyReLU(0.2)
    elif activation == "tanh":
        af = nn.Tanh()
    else:
        raise ValueError(f"unimplemented activation function {activation}")
    # 3 baic convblocks    
    res = ConvBlock(img_channels, h_channels, activation=activation, bn=batch_norm)
    res += ConvBlock(h_channels, h_channels*2, activation=activation, bn=batch_norm)
    res += ConvBlock(h_channels*2, h_channels*4, activation=activation, bn=batch_norm)
    # append 4th conv block if needed
    if n_layers == 4:
        res = res + ConvBlock(h_channels*4,h_channels*4, activation=activation, bn=batch_norm, stride=1)
    elif n_layers != 3:
        raise ValueError(f"this function is implemented only for 3 or 4 layers, not for {n_layers}")
    # now append the reshaping and linear layers
    res.append(Reshape(*(-1, lin_dim)))
    res.append(nn.Linear(lin_dim, z_dim*2))
    res.append(af)
    return nn.Sequential(*res)

def Discriminator(img_channels, h_channels, img_dim, activation="leakyrelu", batch_norm=True, n_layers=3,
    last_sigmoid=True):
    out_dim = img_dim // 8
    lin_dim = h_channels*4*out_dim*out_dim
    res = [# this has to be like this otherwise there are problems with the fm loss
                *ConvBlock(img_channels, h_channels, activation=activation, bn=False, bias=True),
                *ConvBlock(h_channels, h_channels*2, activation=activation, bn=batch_norm),
                *ConvBlock(h_channels*2, h_channels*4, activation=activation, bn=batch_norm)
            ]
    # append 4th conv block if needed
    if n_layers == 4:
        res = res + ConvBlock(h_channels*4,h_channels*4, activation=activation, bn=batch_norm, stride=1)
    elif n_layers != 3:
        raise ValueError(f"this function is implemented only for 3 or 4 layers, not for {n_layers}")
    # now append the reshaping and linear layers
    res.append(Reshape(*(-1, lin_dim)))
    res.append(nn.Linear(lin_dim, 1))
    if last_sigmoid:
        res.append(Sigmoid())

    return nn.Sequential(*res)

def TextureDecoder(z_dim, img_channels, h_channels, init_sz, activation="leakyrelu", batch_norm=True, 
    n_layers=3):
    # first few layers
    res = [        
        nn.Linear(z_dim, h_channels*2 * init_sz ** 2),
        Reshape(*(-1, h_channels*2, init_sz, init_sz)),
        *UpsampleBlock(h_channels*2, h_channels*2, activation=activation, bn=batch_norm),
        *UpsampleBlock(h_channels*2, h_channels, activation=activation, bn=batch_norm)
        ]
    # add another 
    if n_layers == 4:
        res = res + UpsampleBlock(h_channels, h_channels, scale_factor=1, activation=activation,
            bn=batch_norm, stride=1)
    elif n_layers != 3:
        raise ValueError(f"this function is implemented only for 3 or 4 layers, not for {n_layers}")
    # now final convolution and bn
    res.append(nn.Conv2d(h_channels, img_channels, 3, stride=1, padding=1))
    res.append(nn.BatchNorm2d(img_channels)) if batch_norm else None
    res.append(nn.Tanh())
    return nn.Sequential(*res)

def ShapeDecoder(z_dim, img_channels, h_channels, init_sz, activation="leakyrelu", batch_norm=True, 
    n_layers=3):
    # define instance norm
    inst = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    # first few layers
    res = [        
        nn.Linear(z_dim, h_channels*2 * init_sz ** 2),
        Reshape(*(-1, h_channels*2, init_sz, init_sz))
        ]
    res.append(inst(h_channels*2)) if batch_norm else None
    res += UpsampleBlock(h_channels*2, h_channels, activation=activation, bn=False)
    res.append(inst(h_channels)) if batch_norm else None
    # add 4th layer if needed
    if n_layers == 4:
        res += UpsampleBlock(h_channels, h_channels, scale_factor=1, activation=activation, bn=False)
        res.append(inst(h_channels)) if batch_norm else None
    elif n_layers != 3:
        raise ValueError(f"this function is implemented only for 3 or 4 layers, not for {n_layers}")
    # and finally the rest
    # no tanh in the last layers since we want outputs in [0,1]
    res += UpsampleBlock(h_channels, img_channels, activation="leakyrelu", bn=False)
    res.append(inst(img_channels)) if batch_norm else None
    return nn.Sequential(*res)

class Sigmoid(nn.Module):
    def __init__(self, *args):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)

def rp_trick(mu, std):
    """Reparametrization trick via Normal distribution."""
    p = torch.distributions.Normal(mu, std)
    return p.rsample()

def logpx(x, mu, std):
    """Normal log prob."""
    p = torch.distributions.Normal(mu, std)
    dims = tuple(range(1,len(x.shape)))
    return p.log_prob(x).sum(dims)

def batched_score(scoref, loader, device, *args, **kwargs):
    scores = []
    for batch in loader:
        x = batch['ims'].to(device)
        score = scoref(x, *args, **kwargs)
        scores.append(score)

    return np.concatenate(scores, -1)

def create_score_loader(X, batch_size, workers=1, shuffle=False):
    # create the loader
    y = torch.zeros(X.shape[0]).long()
    loader = DataLoader(Subset(torch.tensor(X).float(), y), 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=workers)
    return loader

def all_equal_params(m1, m2):
    """Returns True if all parameters of the two models are the same (does not check for device)."""
    ps = m1.parameters()
    _ps = m2.parameters()
    for (p, _p) in zip(ps, _ps):
        if np.all(p.detach().to('cpu').numpy() != _p.detach().to('cpu').numpy()):
            return False
    return True

def all_nonequal_params(m1, m2):
    """Returns True if all parameters of the two models are different (does not check for device)."""
    ps = m1.parameters()
    _ps = m2.parameters()
    for (p, _p) in zip(ps, _ps):
        if np.any(p.detach().to('cpu').numpy() == _p.detach().to('cpu').numpy()):
            return False
    return True

def get_float(t):
    return float(t.data.cpu().numpy())

# subsample both classes the same
def subsample_same(X,y,n):
    # subsample it - both classes the same amount
    n1 = sum(y).astype('int')
    n0 = (len(y) - n1).astype('int')
    nc = np.minimum(np.minimum(np.floor(n / 2).astype('int'), n1), n0)
    inds0 = np.array(random.sample(range(n0), nc))
    inds1 = np.array(random.sample(range(n1), nc))
    X_sub = np.concatenate((X[y == 0][inds0], X[y == 1][inds1]), 0)
    y_sub = np.concatenate((y[y == 0][inds0], y[y == 1][inds1]), 0)
    return X_sub, y_sub