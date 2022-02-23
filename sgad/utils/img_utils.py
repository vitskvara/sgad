import torch
from torchvision.transforms import ToPILImage
import torch.nn.functional as F

def to_img(x, zoom=1, denormalize=False):
    """Convert a 3D/4D array of (DxHxW)/(NxDxHxW) to a PILimage format. zoom can be int or tuple."""
    if denormalize:
        x = x*0.5+0.5
    if type(zoom) == int:
        zoom = (zoom, zoom)
    t = ToPILImage()
    _x = torch.tensor(x)
    if len(x.shape) == 4:
        _x=torch.concat([_x[i,:,:,:] for i in range(_x.shape[0])],2)
    _x = _x.reshape(1, *_x.shape)
    _x = F.interpolate(_x, (zoom[0]*_x.shape[2], zoom[1]*_x.shape[3]))
    return t(_x.reshape(*_x.shape[1:]))