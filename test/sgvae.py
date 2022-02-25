import unittest
import sgad
import numpy as np
import os
import torch 
import shutil
import time

from sgad.sgvae import SGVAE
from sgad.utils import load_wildlife_mnist, to_img, compute_auc

X_raw, y_raw = load_wildlife_mnist(denormalize=False)

class TestConstructor(unittest.TestCase):
    def _test_vae(self, vae, shape=False):
        self.assertTrue(vae.encoder[0].in_channels == 3)
        self.assertTrue(vae.decoder[0].in_features == 32)
        self.assertTrue(vae.mu_net_z.in_features == 64)
        self.assertTrue(vae.mu_net_z.out_features == 32)
        self.assertTrue(vae.log_var_net_z.in_features == 64)
        self.assertTrue(vae.log_var_net_z.out_features == 32)
        if shape:
        	decoder_out_c = 2
        else:
        	decoder_out_c = 4
        self.assertTrue(vae.mu_net_x.in_channels == decoder_out_c)
        self.assertTrue(vae.mu_net_x.out_channels == decoder_out_c-1)
        self.assertTrue(vae.log_var_net_x[0].in_channels == decoder_out_c)
        self.assertTrue(vae.log_var_net_x[0].out_channels == 1)

    def test_default(self):
        model = SGVAE()
        self.assertTrue(model.num_params() > 5000)
        self.assertTrue(len(next(iter(model.parameters()))) > 0)
        self._test_vae(model.vae_shape, shape=True)
        self._test_vae(model.vae_background)
        self._test_vae(model.vae_foreground)

        self.assertTrue(all(np.array(model.generate(2).shape) == [2, 3, 32, 32]))
        self.assertTrue(all(np.array(model.generate_mean(2).shape) == [2, 3, 32, 32]))
        self.assertTrue(model.opts._modules['sgvae'].defaults['lr'] == 0.0002)
        self.assertTrue(model.opts._modules['sgvae'].defaults['betas'] == [0.5, 0.999])

        self.assertTrue(model.config.lambda_mask == 0.3)
        self.assertTrue(model.config.weight_mask == 100.0)        
        self.assertTrue(model.config.z_dim == 32)
        self.assertTrue(model.config.h_channels == 32)
        self.assertTrue(model.config.img_dim == 32)
        self.assertTrue(model.config.img_channels == 3)
        self.assertTrue(model.config.batch_size == 32) 
        self.assertTrue(model.config.init_type == "orthogonal")
        self.assertTrue(model.config.init_gain == 0.1)
        self.assertTrue(model.config.init_seed == None)
        self.assertTrue(model.config.std_approx == "exp")
        self.assertTrue(model.config.lr == 0.0002)
        self.assertTrue(model.config.betas == [0.5, 0.999])
        
        # test random init
        get_w = lambda model : float(model.vae_shape.encoder[0].weight[0,0,0,0].to('cpu'))
        a = get_w(model)
        model = SGVAE()
        b = get_w(model)
        self.assertTrue(a != b)
        model = SGVAE(init_seed=3)
        a = get_w(model)
        model = SGVAE(init_seed=3)
        b = get_w(model)
        self.assertTrue(a == b)

# test saving weights

class TestFit(unittest.TestCase):
    def test_fit_default(self):
        nc = 0
        X = X_raw[y_raw == nc]
        model = SGVAE(img_dim=X.shape[2], img_channels=X.shape[1], lambda_mask=0.3, weight_mask=240.0)
        _tmp = "./_tmp_sgvae"
        losses_all, best_model, best_epoch = model.fit(X, 
            save_path=_tmp, 
            n_epochs=4)
        shutil.rmtree(_tmp)

