import torch
import numpy as np
from torch import nn
from sgad.cgn.models.cgn import Reshape, UpsampleBlock, lin_block, shape_layers, texture_layers, get_norm_layer
from sklearn.linear_model import LogisticRegression
import warnings

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

def ConvBlock(in_channels, out_channels, bn=True):
    res =  [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)]
    res.append(nn.BatchNorm2d(out_channels)) if bn else None
    res.append(nn.LeakyReLU(0.2))
    return res

def Encoder(z_dim, img_channels, h_channels, img_dim):
    out_dim = img_dim // 8
    lin_dim = h_channels*4*out_dim*out_dim
    return nn.Sequential(
                *ConvBlock(img_channels, h_channels),
                *ConvBlock(h_channels, h_channels*2),
                *ConvBlock(h_channels*2, h_channels*4),
                Reshape(*(-1, lin_dim)),
                nn.Linear(lin_dim, z_dim*2),
                nn.LeakyReLU(0.2)
            )

def TextureDecoder(z_dim, img_channels, h_channels, init_sz):
    return nn.Sequential(*texture_layers(z_dim, img_channels, h_channels, init_sz), nn.Tanh())

def ShapeDecoder(z_dim, img_channels, h_channels, init_sz):
    return nn.Sequential(*shape_layers(z_dim, img_channels, h_channels, init_sz))

class Sigmoid(nn.Module):
    def __init__(self, *args):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)

def Discriminator(img_channels, h_channels, img_dim, first_bn=True):
    out_dim = img_dim // 8
    lin_dim = h_channels*4*out_dim*out_dim
    return nn.Sequential(
                *ConvBlock(img_channels, h_channels, bn=first_bn),
                *ConvBlock(h_channels, h_channels*2),
                *ConvBlock(h_channels*2, h_channels*4),
                Reshape(*(-1, lin_dim)),
                nn.Linear(lin_dim, 1),
                Sigmoid()
            )

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

