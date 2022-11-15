import unittest
import sgad
import numpy as np
import os
import torch 
import shutil
from torch import nn
import copy

from sgad.sgvae import SGVAE
from sgad.utils import load_wildlife_mnist, to_img, compute_auc
from sgad.sgvae.utils import all_equal_params, all_nonequal_params
from sgad.utils import save_cfg, load_cfg, construct_model, load_model

_tmp = "./_tmp_sgvae"
ac = 4
seed = 1
data = sgad.utils.load_wildlife_mnist_split(ac, seed, denormalize = False)
(tr_X, tr_y, tr_c), (val_X, val_y, val_c), (tst_X, tst_y, tst_c) = data

def test_args(X, **kwargs):
    n = 10
    model = SGVAE(**kwargs)
    x = torch.tensor(X[:n]).to(model.device)
    z = torch.randn((n,model.z_dim)).to(model.device)
    xh = model.decode_image((z,z,z))
    zh = model.encoded(x)
    xs = np.array(x.size())
    xs[1] = model.vae_background.out_channels
    return model, (xh.size() == xs).all(), zh[1].size() == (n,model.z_dim)

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

        self.assertTrue(model.config.tau_mask == 0.1)
        self.assertTrue(model.config.weight_mask == 1.0)
        self.assertTrue(model.config.weight_binary == 1.0)
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

        # an advanced test
        model, xo, zo = test_args(tr_X)
        cmodel = copy.deepcopy(model)
        cmodel = model.cpu_copy()
        self.assertTrue(all_equal_params(model, cmodel))
        # make an update
        batch = {'ims': torch.tensor(tr_X[:32])}
        iepoch = 1
        l, elbo, kld, lpx, bin_l, mask_l, text_l, kld_s, kld_b, kld_f = model.train_step_independent(
            batch, iepoch)
        # tests
        self.assertTrue(not all_equal_params(model, cmodel))
        self.assertTrue(not all_nonequal_params(model, cmodel))
        self.assertTrue(not all_nonequal_params(model.vae_shape, cmodel.vae_shape))
        self.assertTrue(all_nonequal_params(model.vae_shape.encoder, cmodel.vae_shape.encoder))
        self.assertTrue(all_nonequal_params(model.vae_shape.decoder, cmodel.vae_shape.decoder))
        self.assertTrue(all_nonequal_params(model.vae_shape.mu_net_z, cmodel.vae_shape.mu_net_z))
        self.assertTrue(all_nonequal_params(model.vae_shape.mu_net_x, cmodel.vae_shape.mu_net_x))
        self.assertTrue(all_nonequal_params(model.vae_shape.log_var_net_z, cmodel.vae_shape.log_var_net_z))
        # because this is not trained
        self.assertTrue(all_equal_params(model.vae_shape.log_var_net_x, cmodel.vae_shape.log_var_net_x))
        self.assertTrue((model.log_var_net_x(1).detach().cpu().numpy() != cmodel.log_var_net_x(1).detach().cpu().numpy())[0])

        # do it again
        model, xo, zo = test_args(tr_X, log_var_x_estimate_top="conv_net")
        cmodel = copy.deepcopy(model)
        cmodel = model.cpu_copy()
        self.assertTrue(all_equal_params(model, cmodel))
        # make an update
        batch = {'ims': torch.tensor(tr_X[:32])}
        iepoch = 1
        l, elbo, kld, lpx, bin_l, mask_l, text_l, kld_s, kld_b, kld_f = model.train_step_independent(
            batch, iepoch)
        # tests
        self.assertTrue(not all_equal_params(model, cmodel))
        self.assertTrue(not all_nonequal_params(model, cmodel))
        self.assertTrue(not all_nonequal_params(model.vae_shape, cmodel.vae_shape))
        self.assertTrue(all_nonequal_params(model.vae_shape.encoder, cmodel.vae_shape.encoder))
        self.assertTrue(all_nonequal_params(model.vae_shape.decoder, cmodel.vae_shape.decoder))
        self.assertTrue(all_nonequal_params(model.vae_shape.mu_net_z, cmodel.vae_shape.mu_net_z))
        self.assertTrue(all_nonequal_params(model.vae_shape.mu_net_x, cmodel.vae_shape.mu_net_x))
        self.assertTrue(all_nonequal_params(model.vae_shape.log_var_net_z, cmodel.vae_shape.log_var_net_z))
        self.assertTrue(all_equal_params(model.vae_shape.log_var_net_x, cmodel.vae_shape.log_var_net_x))

    def test_load_from_cfg(self):
        # first test loading of the config files
        model = SGVAE(h_channels=1)
        cf = "test.yaml"
        save_cfg(model.config, cf)
        cfg = load_cfg(cf)
        self.assertTrue(model.config == cfg)
        model_new = construct_model(SGVAE, cf)
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
class TestFitPredict(unittest.TestCase):
    def test_fit_predict_default(self):
        model = SGVAE(img_dim=tr_X.shape[2], img_channels=tr_X.shape[1], lambda_mask=0.3, weight_mask=240.0,
            weight_binary=100)
        losses_all, best_model, best_epoch = model.fit(tr_X, 
            save_path=_tmp, 
            n_epochs=1,
            workers=4)

        # test basic prediction
        model.eval()
        n = 128
        Xn = tr_X[:n]
        yn = tr_y[:n]
        Xa = val_X[val_y == 1][:n]
        ya = val_y[val_y == 1][:n]
        sn = model.predict(Xn, score_type="logpx", latent_score_type="normal", batch_size=16, workers=4)
        self.assertTrue(len(sn == n))
        sn = model.predict(Xn, n=2, score_type="logpx", latent_score_type="normal", workers=4)
        self.assertTrue(len(sn == n))
        try:
            sn = model.predict(Xn, n=2, score_type="logpx", latent_score_type="normal", probability=True,
                workers=4)
        except Exception as e:
            self.assertTrue(type(e) is ValueError)
        sa = model.predict(Xn, score_type="logpx", latent_score_type="normal", batch_size=16, workers=4)
        self.assertTrue(type(sa) is np.ndarray)
        self.assertTrue(sa.shape == (n,))
        
        # this requires to set n_epochs to a higher number and requires a lot more time
        # self.assertTrue(sa.mean() > sn.mean())

        # test latent scores
        Xnt = torch.tensor(Xn).to(model.device)
        s = model.normal_latent_score(Xnt)
        self.assertTrue(type(s) is np.ndarray)
        self.assertTrue(s.shape == (3, n))
        s = model.all_scores(Xn, score_type="logpx", latent_score_type="normal", batch_size=16, workers=4)
        self.assertTrue(type(s) is np.ndarray)
        self.assertTrue(s.shape == (4, n))

        s = model.kld_score(Xnt)
        self.assertTrue(type(s) is np.ndarray)
        self.assertTrue(s.shape == (3, n))
        s = model.all_scores(Xn, score_type="logpx", latent_score_type="kld", batch_size=16, workers=4)
        self.assertTrue(type(s) is np.ndarray)
        self.assertTrue(s.shape == (4, n))

        s = model.normal_logpx_score(Xnt)
        self.assertTrue(type(s) is np.ndarray)
        self.assertTrue(s.shape == (3, n))
        s = model.all_scores(Xn, score_type="logpx", latent_score_type="normal_logpx", batch_size=16, 
            workers=4)
        self.assertTrue(type(s) is np.ndarray)
        self.assertTrue(s.shape == (4, n))
       
        # fit the goddamn logistic regression
        Xf = val_X[:5000]
        yf = val_y[:5000]
        self.assertTrue(model.alpha is None)
        self.assertTrue(model.alpha_score_type is None)
        model.fit_alpha(Xf, yf, score_type="logpx", latent_score_type="normal", n=2, workers=4)
        self.assertTrue(model.alpha is not None)
        self.assertTrue(model.alpha_score_type == "normal")
        self.assertTrue(len(model.alpha) == 4)
        sn = model.predict(Xn, n=2, score_type="logpx", latent_score_type="normal", probability=True, 
            workers=4)
        sa = model.predict(Xa, n=2, score_type="logpx", latent_score_type="normal", probability=True, 
            workers=4)
        s = np.concatenate((sn, sa))
        y = np.concatenate((yn, ya)).astype('int')
        self.assertTrue(compute_auc(y, s) > 0.5)
        self.assertTrue((0.0 <= s).all() and (s  <= 1.0).all())

        # try the errors
        try:
            sn = model.predict(Xn, n=2, score_type="logpx", latent_score_type="kld", probability=True, 
                workers=4)
        except Exception as e:
            self.assertTrue(type(e) is ValueError)
        try:
            sn = model.predict(Xn, n=2, score_type="logpx", latent_score_type="normal_logpx", probability=True, 
                workers=4)
        except Exception as e:
            self.assertTrue(type(e) is ValueError)
        
        # save/load alphas
        model.save_alpha("_alpha.npy")
        model.load_alpha("_alpha.npy")
        sn = model.predict(Xn, n=2, score_type="logpx", latent_score_type="normal", probability=True, workers=4)
        
        os.remove("_alpha.npy")
        shutil.rmtree(_tmp)

    def test_val_fit(self):
        # construct
        model = SGVAE()

        # fit
        losses_all, best_model, best_epoch = model.fit(tr_X, n_epochs=3, save_path=_tmp, 
            save_weights=True, workers=2, X_val=val_X, y_val=val_y, val_samples=1000)
        best_model.move_to(model.device)
        self.assertTrue(best_epoch >= 1)
        if best_epoch == 3:
            self.assertTrue(all_equal_params(model, best_model))
        else:    
            self.assertTrue(not all_equal_params(model, best_model))

        # scores
        score = model.predict(tst_X, workers=2, n=5)
        auc = compute_auc(tst_y, score)
        self.assertTrue(auc > 0.5)

        # cleanup
        shutil.rmtree(_tmp)

class TestParams(unittest.TestCase):
    def test_params(self):
        X = tr_X[:1000]
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
        model.log_var_x_global.data=torch.tensor([-3]).float().to(model.device)
        self.assertTrue(not all_equal_params(model, _model))
        self.assertTrue(not all_nonequal_params(model, _model))
        # construct the model again and train it for one epoch
        model = SGVAE(img_dim=X.shape[2], img_channels=X.shape[1], log_var_x_estimate="global")
        _model = copy.deepcopy(model)
        model.fit(X,
            n_epochs=2, 
            verb=True, 
            workers=4
           )
        # check the equality of params
        self.assertTrue(not all_equal_params(model, _model))
        self.assertTrue(not all_nonequal_params(model, _model)) # the global log_var_x are not trained
        model.vae_shape.log_var_x_global.data=torch.tensor([-3]).float().to(model.device)
        self.assertTrue(not all_nonequal_params(model, _model))
        model.vae_background.log_var_x_global.data=torch.tensor([-3]).float().to(model.device)
        self.assertTrue(not all_nonequal_params(model, _model))
        model.vae_foreground.log_var_x_global.data=torch.tensor([-3]).float().to(model.device)
        # now the rest of the params must have changed
        #self.assertTrue(all_nonequal_params(model, _model))
        self.assertTrue(not model.vae_shape.log_var_x_global.data == _model.vae_shape.log_var_x_global.data)
        self.assertTrue(all_nonequal_params(model.vae_shape.encoder, _model.vae_shape.encoder))
        self.assertTrue(all_nonequal_params(model.vae_shape.decoder, _model.vae_shape.decoder))
        self.assertTrue(all_nonequal_params(model.vae_shape.mu_net_z, _model.vae_shape.mu_net_z))
        self.assertTrue(all_nonequal_params(model.vae_shape.log_var_net_z, _model.vae_shape.log_var_net_z))
        self.assertTrue(all_nonequal_params(model.vae_shape.mu_net_x, _model.vae_shape.mu_net_x))
