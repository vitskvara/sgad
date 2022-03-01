import torch
import numpy as np
from torch import nn
from sgad.cgn.models.cgn import Reshape, UpsampleBlock, lin_block, shape_layers, texture_layers
from sgad.cgn.models.cgn import get_norm_layer

class Mean(nn.Module):
    def __init__(self, *args):
        super(Mean, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.mean(self.shape) 

def ConvBlock(in_channels, out_channels):
    return [
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    ]

def Encoder(z_dim, img_channels, h_channels, img_dim):
    out_dim = img_dim // 8
    lin_dim = h_channels*4*out_dim*out_dim
    return nn.Sequential(
                *ConvBlock(img_channels, h_channels),
                *ConvBlock(h_channels, h_channels*2),
                *ConvBlock(h_channels*2, h_channels*4),
                Reshape(*(-1, lin_dim)),
                nn.Linear(lin_dim, z_dim*2),
                nn.LeakyReLU(0.2, inplace=True)
            )

def TextureDecoder(z_dim, img_channels, h_channels, init_sz):
    return nn.Sequential(*texture_layers(z_dim, img_channels, h_channels, init_sz), nn.Tanh())

def ShapeDecoder(z_dim, img_channels, h_channels, init_sz):
    return nn.Sequential(*shape_layers(z_dim, img_channels, h_channels, init_sz))

def rp_trick(mu, std):
    """Reparametrization trick via Normal distribution."""
    p = torch.distributions.Normal(mu, std)
    return p.rsample()

def logpx(x, mu, std):
    """Normal log prob."""
    p = torch.distributions.Normal(mu, std)
    return p.log_prob(x).sum((1,2,3))

def batched_score(scoref, loader, device, *args, **kwargs):
    scores = []
    labels = []
    for batch in loader:
        x = batch['ims'].to(device)
        score = scoref(x, *args, **kwargs)
        scores.append(score)

    return np.concatenate(scores)

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

