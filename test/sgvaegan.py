import unittest
import sgad
import numpy as np
import os
import torch 
import shutil
from torch import nn

from sgad.sgvae import SGVAEGAN
from sgad.utils import load_wildlife_mnist, to_img, compute_auc
from sgad.sgvae.utils import all_equal_params, all_nonequal_params
from sgad.utils import save_cfg, load_cfg, construct_model, load_model

_tmp = "./_tmp_sgvaegan"
ac = 4
seed = 1
data = sgad.utils.load_wildlife_mnist_split(ac, seed, denormalize = False)
(tr_X, tr_y, tr_c), (val_X, val_y, val_c), (tst_X, tst_y, tst_c) = data

def test_args(X, **kwargs):
    n = 10
    model = SGVAEGAN(**kwargs)
    x = torch.tensor(X[:n]).to(model.device)
    z = torch.randn((n,model.z_dim)).to(model.device)
    xh = model.decode_image((z,z,z))
    zh = model.encoded(x)
    xs = np.array(x.size())
    xs[1] = model.sgvae.vae_background.out_channels
    return model, (xh.size() == xs).all(), zh[1].size() == (n,model.z_dim)

class TestAll(unittest.TestCase):
    def test_default(self):
        # construct
        model = SGVAEGAN(fm_alpha=10.0, z_dim=128, h_channels=128, fm_depth=7, batch_size=64, 
                       input_range=[-1, 1], weight_texture=100.0)

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
        model_new = load_model(SGVAEGAN, _tmp)
        self.assertTrue(model.config == model_new.config)
        all_equal_params(model, model_new)
        shutil.rmtree(_tmp)

    def test_cpu_copy(self):
        # construct and copy the model
        model, xo, zo = test_args(tr_X)
        self.assertTrue(xo)
        self.assertTrue(zo)
        cmodel = model.cpu_copy()
        cmodel.move_to(model.device)
        self.assertTrue(all_equal_params(model, cmodel))

        x = torch.tensor(tr_X[:64]).to(model.device)
        # update encoders
        z_s, z_b, z_f, x_rec, kld, bin_l, mask_l, text_l, fml, el, kld_s, kld_b, kld_f = model.update_encoders(x, 1)
        self.assertTrue(not all_equal_params(model, cmodel))
        self.assertTrue(not all_nonequal_params(model, cmodel))
        self.assertTrue(all_nonequal_params(model.params.encoders, cmodel.params.encoders))
        self.assertTrue(all_equal_params(model.params.decoders, cmodel.params.decoders))
        self.assertTrue(all_equal_params(model.params.discriminator, cmodel.params.discriminator))
        # update encoders
        x_rec, x_gen, bin_l, mask_l, text_l, fml, gl, dl = model.update_decoders(x, z_s, z_b, z_f, 1)
        self.assertTrue(not all_equal_params(model, cmodel))
        self.assertTrue(not all_nonequal_params(model, cmodel))
        self.assertTrue(all_nonequal_params(model.params.encoders, cmodel.params.encoders))
        self.assertTrue(all_nonequal_params(model.params.decoders, cmodel.params.decoders))
        self.assertTrue(all_equal_params(model.params.discriminator, cmodel.params.discriminator))
        # update discriminator
        discl = model.update_discriminator(x, x_rec, x_gen)
        self.assertTrue(not all_equal_params(model, cmodel))
        self.assertTrue(not all_nonequal_params(model, cmodel))
        self.assertTrue(all_nonequal_params(model.params.encoders, cmodel.params.encoders))
        self.assertTrue(all_nonequal_params(model.params.decoders, cmodel.params.decoders))
        self.assertTrue(all_nonequal_params(model.params.discriminator, cmodel.params.discriminator))

    def test_val_fit(self):
        # construct
        model = SGVAEGAN(fm_alpha=10.0, z_dim=128, h_channels=128, fm_depth=7, batch_size=64, 
                       input_range=[-1, 1], weight_texture=100.0)

        # fit
        losses_all, best_model, best_epoch = model.fit(tr_X, n_epochs=3, save_path=_tmp, save_weights=True, 
            workers=2, X_val = val_X, y_val = val_y, val_samples=1000)
        best_model.move_to(model.device)

        # 
        if best_epoch == 2:
            self.assertTrue(all_equal_params(model, best_model))
        else:
            self.assertTrue(not all_nonequal_params(model, best_model))
        self.assertTrue(model.best_score_type is not None)
        self.assertTrue(model.best_score_type == best_model.best_score_type)

        # 
        disc_score = best_model.predict(val_X, score_type="discriminator", workers=2)
        rec_score = best_model.predict(val_X, score_type="reconstruction", workers=2, n=5)
        fm_score = best_model.predict(val_X, score_type="feature_matching", workers=2, n=5)
        disc_auc = compute_auc(val_y, disc_score)
        rec_auc = compute_auc(val_y, rec_score)
        fm_auc = compute_auc(val_y, fm_score)
        if best_model.best_score_type == "discriminator":
            self.assertTrue(disc_auc > rec_auc)
            self.assertTrue(disc_auc > fm_auc)
        elif best_model.best_score_type == "reconstruction":
            self.assertTrue(rec_auc > disc_auc)
            self.assertTrue(rec_auc > fm_auc)
        elif best_model.best_score_type == "feature_matching":
            self.assertTrue(fm_auc > disc_auc)
            self.assertTrue(fm_auc > rec_auc)

        # cleanup
        shutil.rmtree(_tmp)
