import unittest
import sgad
import numpy as np
from torch.utils.data import DataLoader
import os
import torch 
import shutil

from sgad.utils import load_cifar10
from sgad.cgn import Subset
from sgad.cgn.models import CGNAnomaly

class TestConstructor(unittest.TestCase):
    def test_default(self):
        model = CGNAnomaly()
        self.assertTrue(len(next(iter(model.parameters()))) > 0)
        self.assertTrue(model.cgn.f_shape[0].in_features == 33)
        self.assertTrue(model.config.n_classes == 1)
        self.assertTrue(model.cgn.n_classes == 1)
        self.assertTrue(model.cgn.f_shape[4].out_channels == 32)
        self.assertTrue(all(np.array(model.generate_random(2).shape) == [2, 3, 32, 32]))
        self.assertTrue(type(model.discriminator.model[0]) == torch.nn.modules.linear.Linear)
        self.assertTrue(model.discriminator.model[0].out_features == 32)
        self.assertTrue(model.config.batch_size == 1)
        self.assertTrue(model.config.init_type == 'orthogonal')
        self.assertTrue(model.config.init_gain == 0.1)
        self.assertTrue(model.config.lambda_mask == 0.3)
        self.assertTrue(model.binary_loss.loss_weight == 0.3)
        self.assertTrue(model.config.lambdas_perc == [0.01, 0.05, 0.0, 0.01])
        self.assertTrue(model.perc_loss.model.style_wgts == [0.01, 0.05, 0.0, 0.01])
        self.assertTrue(model.opts._modules['generator'].defaults['lr'] == 0.0002)
        self.assertTrue(model.opts._modules['discriminator'].defaults['lr'] == 0.0002)
        self.assertTrue(model.opts._modules['generator'].defaults['betas'] == [0.5, 0.999])
        self.assertTrue(model.opts._modules['discriminator'].defaults['betas'] == [0.5, 0.999])

    def test_z_dim(self):
        model = CGNAnomaly(z_dim = 16)
        self.assertTrue(model.cgn.f_shape[0].in_features == 17)

    def test_h_channels(self):
        model = CGNAnomaly(h_channels = 8)
        self.assertTrue(model.cgn.f_shape[4].out_channels == 8)

    def test_n_classes(self):
        model = CGNAnomaly(n_classes = 2)
        self.assertTrue(model.config.n_classes == 2)
        self.assertTrue(model.cgn.n_classes == 2)
        self.assertTrue(all(np.array(model.generate_random(2).shape) == [2, 3, 32, 32]))
        
    def test_img_dim(self):
        model = CGNAnomaly(img_dim = 16)
        x_gen = model.generate_random(2)
        self.assertTrue(all(np.array(x_gen.shape) == [2, 3, 16, 16]))
        self.assertTrue(all(np.array(model.generate_random(2).shape) == [2, 3, 16, 16]))

    def test_disc_model(self):
        model = CGNAnomaly(disc_model = 'conv')
        self.assertTrue(type(model.discriminator.model[0]) == torch.nn.modules.conv.Conv2d)

    def test_disc_h_dim(self):
        model = CGNAnomaly(disc_h_dim = 16)
        self.assertTrue(model.discriminator.model[0].out_features == 16)
        model = CGNAnomaly(disc_h_dim = 16, disc_model = 'conv')
        self.assertTrue(model.discriminator.model[0].out_channels == 16)

    def test_batch_size(self):
        model = CGNAnomaly(batch_size = 32)
        self.assertTrue(model.config.batch_size == 32)

    def test_init_type(self):
        model = CGNAnomaly(init_type = 'normal')
        self.assertTrue(model.config.init_type == 'normal')
        model = CGNAnomaly(init_type = 'xavier')
        self.assertTrue(model.config.init_type == 'xavier')
        model = CGNAnomaly(init_type = 'kaiming')
        self.assertTrue(model.config.init_type == 'kaiming')

    def test_init_gain(self):
        # scaling factor for normal, xavier and orthogonal initialization
        model = CGNAnomaly(init_gain = 0.05)
        self.assertTrue(model.config.init_gain == 0.05)

    def test_lambda_mask(self):
        model = CGNAnomaly(lambda_mask = 0.2)
        self.assertTrue(model.config.lambda_mask == 0.2)
        self.assertTrue(model.binary_loss.loss_weight == 0.2)

    def test_lambdas_perc(self):
        model = CGNAnomaly(lambdas_perc = [0.02, 0.1, 0.1, 0.05])
        self.assertTrue(model.config.lambdas_perc == [0.02, 0.1, 0.1, 0.05])
        self.assertTrue(model.perc_loss.model.style_wgts == [0.02, 0.1, 0.1, 0.05])

    def test_lr(self):
        model = CGNAnomaly(lr = 0.001)
        self.assertTrue(model.opts._modules['generator'].defaults['lr'] == 0.001)
        self.assertTrue(model.opts._modules['discriminator'].defaults['lr'] == 0.001)

    def test_betas(self):
        model = CGNAnomaly(betas = [0.48, 0.99])
        self.assertTrue(model.opts._modules['generator'].defaults['betas'] == [0.48, 0.99])
        self.assertTrue(model.opts._modules['discriminator'].defaults['betas'] == [0.48, 0.99])

X_raw, y_raw = load_cifar10()

class TestUtils(unittest.TestCase):
    def test_generate_random(self):
        model = CGNAnomaly()
        x_gen = model.generate_random(4)        
        self.assertTrue(all(np.array(x_gen.shape) == [4, 3, 32, 32]))

    def test_tr_loader(self):
        X = torch.tensor(X_raw)
        y = torch.zeros(X_raw.shape[0]).long()
        batch_size = 32
        tr_loader = DataLoader(Subset(X, y), 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=12)
        batch = next(iter(tr_loader))

        self.assertTrue(all(np.array(batch['ims'].shape) == [batch_size, 3, 32, 32]))
        self.assertTrue(len(batch['labels']) == batch_size)

    def test_save_sample_images(self):
        model = CGNAnomaly()
        _tmp = "_tmp_samples"
        model.save_sample_images(_tmp, 1, n_rows=3)
        self.assertTrue(os.path.isfile(f"{_tmp}/0_1_x_gen.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/1_1_mask.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/2_1_foreground.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/3_1_background.png"))
        shutil.rmtree(_tmp)
