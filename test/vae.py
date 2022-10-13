import unittest
import sgad
import numpy as np
import os
import torch 
import shutil
import time
import copy

from sgad.sgvae import VAE
from sgad.utils import load_wildlife_mnist, to_img, compute_auc
from sgad.sgvae.utils import all_equal_params, all_nonequal_params

X_raw, y_raw = load_wildlife_mnist(denormalize=False)

def test_args(X_raw, **kwargs):
    n = 10
    model = VAE(**kwargs)
    x = torch.tensor(X_raw[:n]).to(model.device)
    z = torch.randn((n,model.z_dim)).to(model.device)
    xh = model.decoded(z)
    zh = model.encoded(x)
    xs = np.array(x.size())
    xs[1] = model.out_channels
    return model, (xh.size() == xs).all(), zh.size() == (n,model.z_dim)

class TestConstructor(unittest.TestCase):
    def test_default(self):
        model, xo, zo = test_args(X_raw)
        self.assertTrue(xo)
        self.assertTrue(zo)
        self.assertTrue(model.num_params() > 5000)
        self.assertTrue(len(next(iter(model.parameters()))) > 0)
        self.assertTrue(model.encoder[0].in_channels == 3)
        self.assertTrue(model.decoder[0].in_features == 32)
        self.assertTrue(model.mu_net_z.in_features == 64)
        self.assertTrue(model.mu_net_z.out_features == 32)
        self.assertTrue(model.log_var_net_z.in_features == 64)
        self.assertTrue(model.log_var_net_z.out_features == 32)
        self.assertTrue(model.mu_net_x.in_channels == 4)
        self.assertTrue(model.mu_net_x.out_channels == 3)
        self.assertTrue(model.log_var_net_x[0].in_channels == 4)
        self.assertTrue(model.log_var_net_x[0].out_channels == 1)

        self.assertTrue(all(np.array(model.generate(2).shape) == [2, 3, 32, 32]))
        self.assertTrue(all(np.array(model.generate_mean(2).shape) == [2, 3, 32, 32]))
        self.assertTrue(model.opts._modules['vae'].defaults['lr'] == 0.0002)
        self.assertTrue(model.opts._modules['vae'].defaults['betas'] == [0.5, 0.999])
        
        self.assertTrue(model.config.z_dim == 32)
        self.assertTrue(model.config.h_channels == 32)
        self.assertTrue(model.config.img_dim == 32)
        self.assertTrue(model.config.img_channels == 3)
        self.assertTrue(model.config.n_layers == 3)
        self.assertTrue(model.config.activation == "leakyrelu")
        self.assertTrue(model.config.batch_norm == True)
        self.assertTrue(model.config.batch_size == 32) 
        self.assertTrue(model.config.init_type == "orthogonal")
        self.assertTrue(model.config.init_gain == 0.1)
        self.assertTrue(model.config.init_seed == None)
        self.assertTrue(model.config.vae_type == "texture")
        self.assertTrue(model.config.std_approx == "exp")
        self.assertTrue(model.config.optimizer == "adam")
        self.assertTrue(model.config.lr == 0.0002)
        self.assertTrue(model.config.betas == [0.5, 0.999])
        
        # test random init
        get_w = lambda model : float(model.encoder[0].weight[0,0,0,0].to('cpu'))
        a = get_w(model)
        model = VAE()
        b = get_w(model)
        self.assertTrue(a != b)
        model = VAE(init_seed=3)
        a = get_w(model)
        model = VAE(init_seed=3)
        b = get_w(model)
        self.assertTrue(a == b)

    def test_constructor(self):
        model, xo, zo = test_args(X_raw, vae_type = "shape")
        self.assertTrue(xo)
        self.assertTrue(zo)

        model, xo, zo = test_args(X_raw, n_layers=4)
        self.assertTrue(xo)
        self.assertTrue(zo)
        self.assertTrue(len(model.encoder) == 15)
        self.assertTrue(len(model.decoder) == 17)

        model, xo, zo = test_args(X_raw, n_layers=4, batch_norm=False)
        self.assertTrue(xo)
        self.assertTrue(zo)
        self.assertTrue(len(model.encoder) == 11)
        self.assertTrue(len(model.decoder) == 13)

        model, xo, zo = test_args(X_raw, n_layers=3, batch_norm=False)
        self.assertTrue(xo)
        self.assertTrue(zo)
        self.assertTrue(len(model.encoder) == 9)
        self.assertTrue(len(model.decoder) == 10)

        model, xo, zo = test_args(X_raw, n_layers=3, batch_norm=False, vae_type="shape")
        self.assertTrue(xo)
        self.assertTrue(zo)
        self.assertTrue(len(model.encoder) == 9)
        self.assertTrue(len(model.decoder) == 8)

        model, xo, zo = test_args(X_raw, n_layers=3, batch_norm=False, vae_type="shape", activation="tanh")
        self.assertTrue(xo)
        self.assertTrue(zo)
        self.assertTrue(len(model.encoder) == 9)
        self.assertTrue(len(model.decoder) == 8)
        self.assertTrue(all(model.decoder[4](torch.Tensor([3000])) == torch.Tensor([1])))

    def test_shape(self):
        model = VAE(vae_type="shape")
        self.assertTrue(model.num_params() > 5000)
        self.assertTrue(len(next(iter(model.parameters()))) > 0)
        self.assertTrue(model.encoder[0].in_channels == 3)
        self.assertTrue(model.decoder[0].in_features == 32)
        self.assertTrue(model.decoder[-3].out_channels == 2)
        self.assertTrue(model.mu_net_z.in_features == 64)
        self.assertTrue(model.mu_net_z.out_features == 32)
        self.assertTrue(model.log_var_net_z.in_features == 64)
        self.assertTrue(model.log_var_net_z.out_features == 32)
        self.assertTrue(model.mu_net_x.in_channels == 2)
        self.assertTrue(model.mu_net_x.out_channels == 1)
        self.assertTrue(model.log_var_net_x[0].in_channels == 2)
        self.assertTrue(model.log_var_net_x[0].out_channels == 1)

        self.assertTrue(all(np.array(model.generate(2).shape) == [2, 1, 32, 32]))
        self.assertTrue(all(np.array(model.generate_mean(2).shape) == [2, 1, 32, 32]))
        
        self.assertTrue(model.config.vae_type == "shape")

class TestUtils(unittest.TestCase):
    def test_cpu_copy(self):
        model = VAE()
        device = model.device
        cpu_model = model.cpu_copy()
        # make sure that the device does not change after the copy
        self.assertTrue(device.type == next(model.parameters()).device.type)
        if device.type != 'cuda':
            self.assertTrue(device.type == next(cpu_model.parameters()).device.type)
        self.assertTrue(all_equal_params(model, cpu_model))
        model.train_step(torch.tensor(X_raw[:10]).to(model.device))
        w = model.encoder[0].weight[0,0]
        cpuw = cpu_model.encoder[0].weight[0,0]
        self.assertTrue(not (cpuw.to(model.device) == w).any())

# test that all the parts are trained
# test the different constructors
# rewrite save_stuff

_tmp = "./_tmp_vae"
class TestFit(unittest.TestCase):
    def test_fit_default(self):
        nc = 0
        X = X_raw[y_raw == nc]
        model = VAE(img_dim=X.shape[2], img_channels=X.shape[1])
        x = torch.tensor(X[:10]).to(model.device)
        _, _, elbo, _, _, _, _ = model.train_step(x)
        model.fit(X,
            n_epochs=10, 
            save_iter=1000, 
            verb=True, 
            save_path=_tmp
           )
        _, _, _elbo, _, _, _, _ = model.train_step(x)
        self.assertTrue(elbo > _elbo)
        shutil.rmtree(_tmp)
        
    def test_fit_shape(self):
        nc = 0
        X = X_raw[y_raw == nc][:,0:1,:,:]
        model = VAE(img_dim=X.shape[2], img_channels=X.shape[1], vae_type="shape")
        x = torch.tensor(X[:10]).to(model.device)
        _, _, elbo, _, _, _, _ = model.train_step(x)
        model.fit(X,
            n_epochs=10, 
            save_iter=1000, 
            verb=True, 
            save_path=_tmp
           )
        _, _, _elbo, _, _, _, _ = model.train_step(x)
        self.assertTrue(elbo > _elbo)
        shutil.rmtree(_tmp)

    def test_fit_rmsprop(self):
        nc = 0
        X = X_raw[y_raw == nc]
        model = VAE(img_dim=X.shape[2], img_channels=X.shape[1], optimizer="rmsprop")
        x = torch.tensor(X[:10]).to(model.device)
        _, _, elbo, _, _, _, _ = model.train_step(x)
        model.fit(X,
            n_epochs=5, 
            save_iter=1000, 
            verb=True, 
            save_path=_tmp
           )
        _, _, _elbo, _, _, _, _ = model.train_step(x)
        self.assertTrue(elbo > _elbo)
        shutil.rmtree(_tmp)

class TestParams(unittest.TestCase):
    def test_params(self):
        nc = 0
        X = X_raw[y_raw == nc]
        model = VAE(img_dim=X.shape[2], img_channels=X.shape[1], log_var_x_estimate="global")
        _model = copy.deepcopy(model)
        # are all the parts equal in terms of trainable params?
        self.assertTrue(all_equal_params(model, _model))
        self.assertTrue(not all_nonequal_params(model, _model))
        self.assertTrue(all_equal_params(model.mu_net_z, _model.mu_net_z))
        self.assertTrue(all_equal_params(model.log_var_net_z, _model.log_var_net_z))
        self.assertTrue(all_equal_params(model.mu_net_x, _model.mu_net_x))
        # now change one param
        model.log_var_x_global.data=torch.tensor([-3]).to(model.device)
        self.assertTrue(not all_equal_params(model, _model))
        self.assertTrue(not all_nonequal_params(model, _model))
        # construct the model again and train it for one epoch
        model = VAE(img_dim=X.shape[2], img_channels=X.shape[1], log_var_x_estimate="global")
        _model = copy.deepcopy(model)
        model.fit(X,
            n_epochs=1, 
            save_iter=1000, 
            verb=True, 
            )
        # check the equality of params
        self.assertTrue(not all_equal_params(model, _model))
        self.assertTrue(all_nonequal_params(model, _model))
        self.assertTrue(all_nonequal_params(model.encoder, _model.encoder))
        self.assertTrue(all_nonequal_params(model.decoder, _model.decoder))
        self.assertTrue(all_nonequal_params(model.mu_net_z, _model.mu_net_z))
        self.assertTrue(all_nonequal_params(model.log_var_net_z, _model.log_var_net_z))
        self.assertTrue(all_nonequal_params(model.mu_net_x, _model.mu_net_x))
