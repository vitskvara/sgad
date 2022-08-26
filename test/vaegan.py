import unittest
import sgad
import numpy as np
import os
import torch 
import shutil
from torch import nn

from sgad.sgvae import VAEGAN
from sgad.utils import load_wildlife_mnist, to_img, compute_auc
from sgad.sgvae.utils import all_equal_params, all_nonequal_params
from sgad.utils import save_cfg, load_cfg, construct_model, load_model

_tmp = "./_tmp_vaegan"
ac = 4
seed = 1
data = sgad.utils.load_wildlife_mnist_split(ac, seed, denormalize = False)
(tr_X, tr_y, tr_c), (val_X, val_y, val_c), (tst_X, tst_y, tst_c) = data

def test_args(X, **kwargs):
    n = 10
    model = VAEGAN(**kwargs)
    x = torch.tensor(X[:n]).to(model.device)
    z = torch.randn((n,model.z_dim)).to(model.device)
    xh = model.decode(z)
    zh = model.encode(x)
    xs = np.array(x.size())
    xs[1] = model.out_channels
    return model, (xh.size() == xs).all(), zh.size() == (n,model.z_dim)

model, xo, zo = test_args(tr_X)

class TestAll(unittest.TestCase):
    def test_default(self):
        # construct
        model = VAEGAN(fm_alpha=10.0, z_dim=128, h_channels=128, fm_depth=7, batch_size=64, 
                       input_range=[-1, 1])

        # fit
        losses_all, _, _ = model.fit(tr_X, n_epochs=3, save_path=_tmp, save_weights=True, workers=2)

        # some prerequisites
        n = 10
        x = torch.Tensor(tr_X[:n]).to(model.device)
        model.eval()
        # generate
        _x = model.generate(n)
        self.assertTrue(_x.shape[0] == n)
        self.assertTrue(_x.shape == x.shape)
        # reconstrut
        _x = model.reconstruct(x)
        self.assertTrue(_x.shape[0] == n)
        self.assertTrue(_x.shape == x.shape)
        
        # scores
        disc_score = model.predict(tst_X, score_type="discriminator", workers=2)
        rec_score = model.predict(tst_X, score_type="reconstruction", workers=2, n=5)
        fm_score = model.predict(tst_X, score_type="feature_matching", workers=2, n=5)
        disc_auc = compute_auc(tst_y, disc_score)
        rec_auc = compute_auc(tst_y, rec_score)
        fm_auc = compute_auc(tst_y, fm_score)
        self.assertTrue(disc_auc > 0.5)
        self.assertTrue(rec_auc > 0.5)
        self.assertTrue(fm_auc > 0.5)

        # check if everything was saved
        self.assertTrue(os.path.isdir(_tmp))
        self.assertTrue(os.path.isfile(os.path.join(_tmp, "cfg.yaml")))
        self.assertTrue(os.path.isfile(os.path.join(_tmp, "losses.csv")))
        self.assertTrue(os.path.isdir(os.path.join(_tmp, "weights")))
        self.assertTrue(os.path.isdir(os.path.join(_tmp, "samples")))
        self.assertTrue(len(os.listdir(os.path.join(_tmp, "weights"))) > 0)
        self.assertTrue(len(os.listdir(os.path.join(_tmp, "samples"))) > 0)

        # model loading
        model_new = load_model(VAEGAN, _tmp)
        self.assertTrue(model.config == model_new.config)
        all_equal_params(model, model_new)
        shutil.rmtree(_tmp)
