import unittest
import sgad
import numpy as np
import os
import torch 
import shutil
import time
import copy
from torch import nn

from sgad.sgvae import SGVAE
from sgad.utils import load_wildlife_mnist, to_img, compute_auc
from sgad.sgvae.utils import all_equal_params, all_nonequal_params
from sgad.utils import save_cfg, load_cfg, construct_model, load_model

_tmp = "./_tmp_sgvae"
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

class TestUtils(unittest.TestCase):
    def test_cpu_copy(self):
        model = SGVAE(h_channels=2)
        _model = model.cpu_copy()
        model.move_to('cpu')
        self.assertTrue(all_equal_params(model, _model))
        self.assertTrue(model.config == _model.config)
        x = torch.ones(1,3,32,32)
        self.assertTrue((model.encode(x)[0][0] == _model.encode(x)[0][0]).all().item())
        zs = [torch.ones(1,32) for _ in range(3)]
        self.assertTrue((model.decode(zs)[2][0] == _model.decode(zs)[2][0]).all().item())

    def test_load_from_cfg(self):
        # first test loading of the config files
        model = SGVAE(h_channels=1)
        cf = "test.yaml"
        save_cfg(model.config, cf)
        cfg = load_cfg(cf)
        self.assertTrue(model.config == cfg)
        model_new = model_from_config(SGVAE, cf)
        self.assertTrue(model.config == model_new.config)
        os.remove(cf)

        # now test model construction
        model_path, sample_path, weights_path = model.setup_paths(_tmp, True, 20, 20, 2)
        cf = os.path.join(_tmp, "cfg.yaml")
        cfg = load_cfg(cf)
        model_new = construct_model(SGVAE, cf)
        self.assertTrue(model.config == model_new.config)

        # loading of the whole model
        wf = os.path.join(weights_path, "200.pth")
        model.save_weights(wf)
        model_new = load_model(SGVAE, _tmp)
        self.assertTrue(model.config == model_new.config)
        self.assertTrue(all_equal_params(model, model_new))

        # test if choosing the iter works
        model.log_var_x_global.data = nn.Parameter(torch.Tensor([-2.0]))
        wf = os.path.join(weights_path, "400.pth")
        model.save_weights(wf)
        model_new = load_model(SGVAE, _tmp)
        self.assertTrue(model.config == model_new.config)
        all_equal_params(model, model_new)
        model_new = load_model(SGVAE, _tmp, 200)
        self.assertTrue(model.config == model_new.config)
        self.assertTrue(not all_equal_params(model, model_new))
        shutil.rmtree(_tmp)

# test saving weights
class TestFit(unittest.TestCase):
    def test_fit_default(self):
        nc = 0
        X = X_raw[y_raw == nc]
        model = SGVAE(img_dim=X.shape[2], img_channels=X.shape[1], lambda_mask=0.3, weight_mask=240.0)
        losses_all, best_model, best_epoch = model.fit(X, 
            save_path=_tmp, 
            n_epochs=4)
        shutil.rmtree(_tmp)

class TestParams(unittest.TestCase):
    def test_params(self):
        nc = 0
        X = X_raw[y_raw == nc]
        model = SGVAE(img_dim=X.shape[2], img_channels=X.shape[1], log_var_x_estimate="global")
        _model = copy.deepcopy(model)
        # are all the parts equal in terms of trainable params?
        self.assertTrue(all_equal_params(model, _model))
        self.assertTrue(not all_nonequal_params(model, _model))
        self.assertTrue(all_equal_params(model.vae_shape, _model.vae_shape))
        self.assertTrue(all_equal_params(model.vae_shape.mu_net_z, _model.vae_shape.mu_net_z))
        self.assertTrue(all_equal_params(model.vae_shape.log_var_net_z, _model.vae_shape.log_var_net_z))
        self.assertTrue(all_equal_params(model.vae_shape.mu_net_x, _model.vae_shape.mu_net_x))
        # now change one param
        model.log_var_x_global.data=torch.tensor([-3]).to(model.device)
        self.assertTrue(not all_equal_params(model, _model))
        self.assertTrue(not all_nonequal_params(model, _model))
        # construct the model again and train it for one epoch
        model = SGVAE(img_dim=X.shape[2], img_channels=X.shape[1], log_var_x_estimate="global")
        _model = copy.deepcopy(model)
        model.fit(X,
            n_epochs=2, 
            save_iter=1000, 
            verb=True, 
            save_results=False
           )
        # check the equality of params
        self.assertTrue(not all_equal_params(model, _model))
        self.assertTrue(not all_nonequal_params(model, _model)) # the global log_var_x are not trained
        model.vae_shape.log_var_x_global.data=torch.tensor([-3]).to(model.device)
        self.assertTrue(not all_nonequal_params(model, _model))
        model.vae_background.log_var_x_global.data=torch.tensor([-3]).to(model.device)
        self.assertTrue(not all_nonequal_params(model, _model))
        model.vae_foreground.log_var_x_global.data=torch.tensor([-3]).to(model.device)
        # now the rest of the params must have changed
        self.assertTrue(all_nonequal_params(model, _model))
        self.assertTrue(all_nonequal_params(model.vae_shape.encoder, _model.vae_shape.encoder))
        self.assertTrue(all_nonequal_params(model.vae_shape.decoder, _model.vae_shape.decoder))
        self.assertTrue(all_nonequal_params(model.vae_shape.mu_net_z, _model.vae_shape.mu_net_z))
        self.assertTrue(all_nonequal_params(model.vae_shape.log_var_net_z, _model.vae_shape.log_var_net_z))
        self.assertTrue(all_nonequal_params(model.vae_shape.mu_net_x, _model.vae_shape.mu_net_x))
