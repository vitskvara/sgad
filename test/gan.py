import unittest
import sgad
import numpy as np
import os
import torch 
import shutil
from torch import nn

from sgad.sgvae import GAN
from sgad.utils import load_wildlife_mnist, to_img, compute_auc
from sgad.sgvae.utils import all_equal_params, all_nonequal_params
from sgad.utils import save_cfg, load_cfg, construct_model, load_model

_tmp = "./_tmp_gan"
ac = 4
seed = 1
data = sgad.utils.load_wildlife_mnist_split(ac, seed, denormalize = False)
(tr_X, tr_y, tr_c), (val_X, val_y, val_c), (tst_X, tst_y, tst_c) = data

def test_params(tr_X, **kwargs):
	n = 10
	model = GAN(**kwargs)
	x = torch.tensor(tr_X[:n]).to(model.device)
	z = torch.randn((n,model.z_dim)).to(model.device)
	xh = model.generator(z)
	s = model.discriminator(x)
	return model, xh.size() == x.size(), s.size() == (n,1)

class TestAll(unittest.TestCase):
	def test_constructor(self):
		# default
		model, xs, os = test_params(tr_X)
		self.assertTrue(xs)
		self.assertTrue(os)
		self.assertTrue(len(model.generator) == 13)
		self.assertTrue(len(model.discriminator) == 11)

		# 4 layers
		model, xs, os = test_params(tr_X, n_layers=4)
		self.assertTrue(xs)
		self.assertTrue(os)
		self.assertTrue(len(model.generator) == 17)
		self.assertTrue(len(model.discriminator) == 14)

		# no bn
		model, xs, os = test_params(tr_X, batch_norm=False)
		self.assertTrue(xs)
		self.assertTrue(os)
		self.assertTrue(len(model.generator) == 10)
		self.assertTrue(len(model.discriminator) == 9)

		# 
		model, xs, os = test_params(tr_X, n_layers=4, batch_norm=False)
		self.assertTrue(xs)
		self.assertTrue(os)
		self.assertTrue(len(model.generator) == 13)
		self.assertTrue(len(model.discriminator) == 11)

		# tanh
		model, xs, os = test_params(tr_X, n_layers=4, batch_norm=False, activation="tanh")
		self.assertTrue(xs)
		self.assertTrue(os)
		self.assertTrue(len(model.generator) == 13)
		self.assertTrue(len(model.discriminator) == 11)
		self.assertTrue(all(model.generator[4](torch.Tensor([3000])) == torch.Tensor([1])))

		# shape
		model, xs, os = test_params(tr_X, gan_type="shape")
		self.assertTrue(xs)
		self.assertTrue(os)
		self.assertTrue(len(model.generator) == 11)
		self.assertTrue(len(model.discriminator) == 11)

		# shape + n_layers
		model, xs, os = test_params(tr_X, gan_type="shape", n_layers=4)
		self.assertTrue(xs)
		self.assertTrue(os)
		self.assertTrue(len(model.generator) == 15)
		self.assertTrue(len(model.discriminator) == 14)

		# shape + n_layers - bn
		model, xs, os = test_params(tr_X, gan_type="shape", n_layers=4, batch_norm=False)
		self.assertTrue(xs)
		self.assertTrue(os)
		self.assertTrue(len(model.generator) == 11)
		self.assertTrue(len(model.discriminator) == 11)

	def test_fit(self):
		# construct
		model = GAN(alpha=10.0, z_dim=128, h_channels=128, fm_depth=7, batch_size=64, 
					   input_range=[-1, 1], optimizer="rmsprop")

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

		# scores
		disc_score = model.predict(tst_X, workers=2)
		disc_auc = compute_auc(tst_y, disc_score)
		self.assertTrue(disc_auc > 0.5)

		# check if everything was saved
		self.assertTrue(os.path.isdir(_tmp))
		self.assertTrue(os.path.isfile(os.path.join(_tmp, "cfg.yaml")))
		self.assertTrue(os.path.isfile(os.path.join(_tmp, "losses.csv")))
		self.assertTrue(os.path.isdir(os.path.join(_tmp, "weights")))
		self.assertTrue(os.path.isdir(os.path.join(_tmp, "samples")))
		self.assertTrue(len(os.listdir(os.path.join(_tmp, "weights"))) > 0)
		self.assertTrue(len(os.listdir(os.path.join(_tmp, "samples"))) > 0)

		# model loading
#		model_new = load_model(VAEGAN, _tmp)
#		self.assertTrue(model.config == model_new.config)
#		all_equal_params(model, model_new)
		shutil.rmtree(_tmp)
