import unittest
import sgad
import numpy as np
import os
import torch 
import shutil
from torch import nn
import copy

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
    xs[1] = model.vae.out_channels
    return model, (xh.size() == xs).all(), zh.size() == (n,model.z_dim)

class TestAll(unittest.TestCase):
    def test_args(self):        
        model, xo, zo = test_args(tr_X)
        self.assertTrue(xo)
        self.assertTrue(zo)
        self.assertTrue(len(model.vae.encoder) == 12)
        self.assertTrue(len(model.vae.decoder) == 13)
        self.assertTrue(len(model.discriminator) == 11)

        model, xo, zo = test_args(tr_X, n_layers=4)
        self.assertTrue(xo)
        self.assertTrue(zo)
        self.assertTrue(len(model.vae.encoder) == 15)
        self.assertTrue(len(model.vae.decoder) == 17)
        self.assertTrue(len(model.discriminator) == 14)

        model, xo, zo = test_args(tr_X, n_layers=4, batch_norm=False)
        self.assertTrue(xo)
        self.assertTrue(zo)
        self.assertTrue(len(model.vae.encoder) == 11)
        self.assertTrue(len(model.vae.decoder) == 13)
        self.assertTrue(len(model.discriminator) == 11)

        model, xo, zo = test_args(tr_X, n_layers=3, batch_norm=False)
        self.assertTrue(xo)
        self.assertTrue(zo)
        self.assertTrue(len(model.vae.encoder) == 9)
        self.assertTrue(len(model.vae.decoder) == 10)
        self.assertTrue(len(model.discriminator) == 9)

        model, xo, zo = test_args(tr_X, n_layers=3, batch_norm=False, vae_type="shape")
        self.assertTrue(xo)
        self.assertTrue(zo)
        self.assertTrue(len(model.vae.encoder) == 9)
        self.assertTrue(len(model.vae.decoder) == 8)
        self.assertTrue(len(model.discriminator) == 9)

        model, xo, zo = test_args(tr_X, n_layers=3, batch_norm=False, vae_type="shape", activation="tanh")
        self.assertTrue(xo)
        self.assertTrue(zo)
        self.assertTrue(len(model.vae.encoder) == 9)
        self.assertTrue(len(model.vae.decoder) == 8)
        self.assertTrue(all(model.vae.decoder[4](torch.Tensor([3000])) == torch.Tensor([1])))
        self.assertTrue(len(model.discriminator) == 9)

    def test_default(self):
        # construct
        model = VAEGAN(fm_alpha=10.0, z_dim=128, h_channels=128, fm_depth=7, batch_size=64)

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


    def test_cpu_copy(self):
        model, xo, zo = test_args(tr_X, log_var_x_estimate="conv_net")
        cmodel = model.cpu_copy()
        cmodel.move_to(model.device)

        x = torch.tensor(tr_X[:32]).to(model.device)
        loss_vals = model.update_step(x)

        self.assertTrue(all_nonequal_params(model.vae.encoder, cmodel.vae.encoder))
        self.assertTrue(all_nonequal_params(model.vae.mu_net_z, cmodel.vae.mu_net_z))
        self.assertTrue(all_nonequal_params(model.vae.log_var_net_x, cmodel.vae.log_var_net_z))
        self.assertTrue(all_nonequal_params(model.vae.decoder, cmodel.vae.decoder))
        self.assertTrue(all_nonequal_params(model.vae.mu_net_x, cmodel.vae.mu_net_x))
        # since this is not trained at all
        self.assertTrue(all_equal_params(model.vae.log_var_net_x, cmodel.vae.log_var_net_x))
        self.assertTrue(all_nonequal_params(model.discriminator, cmodel.discriminator))

