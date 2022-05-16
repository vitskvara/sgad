from contextlib import redirect_stdout
from yacs.config import CfgNode as CN
import os, torch
import sgad

def save_cfg(cfg, path):
    with open(path, 'w') as f:
        with redirect_stdout(f): print(cfg.dump())

def load_cfg(file):
    cfg = CN()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(file)
    return cfg

def construct_model(mc, cf):
    """
    construct_model(model_constructor, config_file) 
    
    Constructs a model with the parameters as saved in the config file.
    """
    cfg = load_cfg(cf)
    return mc(**dict(zip(cfg.keys(), cfg.values())))

def load_model(mc, md, niter=None, device=None, file_descriptor=''):
    """
    load_model(model_constructor, model_dir, niter=None)

    Loads the model from a directory containing model config and weights.
    """
    # first construct model with the saved params
    cf = os.path.join(md, "cfg.yaml")
    model = construct_model(mc, cf)
    if device is not None:
        model.move_to(device)
    # then load weights and replace them
    wdir = os.path.join(md, "weights")
    if niter is None: # load the latest one
        wfs = os.listdir(wdir)
        wfs = list(map(lambda x: int(x.split(".")[0]), wfs))
        niter = max(wfs)
    if not file_descriptor == '':
        file_descriptor = file_descriptor*'_'
    wf = os.path.join(wdir, f'{file_descriptor}{niter}.pth')
    if device is None:
        weights = torch.load(wf)
    else:
        weights = torch.load(wf, map_location=torch.device(device))
    model.load_state_dict(weights)
    return model

def load_cgnanomaly(md, niter=None, device=None):
    """
    load_cgn(md, niter=None, device=None):

    Loads the CGNAnomaly model from a directory containing model config and weights.
    """
    # first construct model with the saved params
    cf = os.path.join(md, "cfg.yaml")
    model = construct_model(sgad.cgn.models.CGNAnomaly, cf)
    if device is not None:
        model.move_to(device)
    # then load weights and replace them
    wdir = os.path.join(md, "weights")
    if niter is None: # load the latest one
        wfs = os.listdir(wdir)
        wfs = list(map(lambda x: int(x.split(".")[0].split("_")[1]), wfs))
        niter = max(wfs)
    cf = os.path.join(wdir, f'cgn_{niter}.pth')
    df = os.path.join(wdir, f'discriminator_{niter}.pth')
    if device is None:
        cweights = torch.load(cf)
        dweights = torch.load(df)
    else:
        cweights = torch.load(cf, map_location=torch.device(device))
        dweights = torch.load(df, map_location=torch.device(device))
    model.cgn.load_state_dict(cweights)
    model.discriminator.load_state_dict(dweights)
    return model
