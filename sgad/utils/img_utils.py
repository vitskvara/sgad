import torch
from torchvision.transforms import ToPILImage
import torch.nn.functional as F

def to_img(x, sz=32):
    """Convert a 3D array of (DxHxW) to a PILimage format."""
    t = ToPILImage()
    _x = torch.tensor(x.reshape(1, *x.shape))
    _x = F.interpolate(_x, (sz,int(sz/x.shape[1]*x.shape[2])))
    return t(_x.reshape(*_x.shape[1:]))