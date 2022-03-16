from contextlib import redirect_stdout
from yacs.config import CfgNode as CN
import os, torch

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

def load_model(mc, md, niter=None):
    """
    load_model(model_constructor, model_dir, niter=None)

    Loads the model from a dirictory containing model config and weights.
    """
    # first construct model with the saved params
    cf = os.path.join(md, "cfg.yaml")
    model = construct_model(mc, cf)
    # then load weights and replace them
    wdir = os.path.join(md, "weights")
    if niter is None: # load the latest one
        wfs = os.listdir(wdir)
        wfs = list(map(lambda x: int(x.split(".")[0]), wfs))
        niter = max(wfs)
    wf = os.path.join(wdir, f'{niter}.pth')
    weights = torch.load(wf)
    model.load_state_dict(weights)
    return model
