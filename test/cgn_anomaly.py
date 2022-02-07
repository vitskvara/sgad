import unittest
import sgad
import numpy as np
from torch.utils.data import DataLoader
import os
import torch 
import shutil

from sgad.utils import load_cifar10, compute_auc
from sgad.cgn import Subset
from sgad.cgn.models import CGNAnomaly

X_raw, y_raw = load_cifar10()

class TestConstructor(unittest.TestCase):
    def test_default(self):
        model = CGNAnomaly()
        self.assertTrue(model.num_params() > 5000)
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
        
        # test random init
        a = float(model.cgn.f_shape[0].weight[0,0].to('cpu'))
        model = CGNAnomaly()
        b = float(model.cgn.f_shape[0].weight[0,0].to('cpu'))
        self.assertTrue(a != b)

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

    def test_img_channels(self):
        model = CGNAnomaly(img_channels = 1)
        x_gen = model.generate_random(2)
        self.assertTrue(all(np.array(x_gen.shape) == [2, 1, 32, 32]))
        self.assertTrue(all(np.array(model.generate_random(2).shape) == [2, 1, 32, 32]))


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

    def test_init_seed(self):
        model = CGNAnomaly(init_seed = 3)
        a = float(model.cgn.f_shape[0].weight[0,0].to('cpu'))
        model = CGNAnomaly(init_seed = 3)
        b = float(model.cgn.f_shape[0].weight[0,0].to('cpu'))
        self.assertTrue(a == b)

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

    def test_cpu_copy(self):
        model = CGNAnomaly()
        device = model.device
        cpu_model = model.cpu_copy()
        # make sure that the device does not change after the copy
        self.assertTrue(device.type == next(model.cgn.parameters()).device.type)
        if device.type != 'cuda':
            self.assertTrue(device.type == next(cpu_model.cgn.parameters()).device.type)
        p1 = next(iter(cpu_model.cgn.parameters())).data.to('cpu')[0]
        p2 = next(iter(model.cgn.parameters())).data.to('cpu')[0]
        self.assertTrue(p1 == p2)
        p1 = next(iter(cpu_model.discriminator.parameters())).data.to('cpu')[0]
        p2 = next(iter(model.discriminator.parameters())).data.to('cpu')[0]
        self.assertTrue(p1 == p2)

class TestFit(unittest.TestCase):
    def test_fit_default(self):
        model = CGNAnomaly(batch_size=32)
        X = X_raw[y_raw==0][:5000]
        _tmp = "./_cgn_anomaly_tmp"
        losses_all, (best_model, best_epoch) = model.fit(
            X, 
            n_epochs=3, 
            save_iter=100, 
            verb=True, 
            save_results=True, 
            save_path=_tmp, 
            workers=12
        )
        self.assertTrue(os.path.isfile(f"{_tmp}/cfg.yaml"))
        self.assertTrue(os.path.isfile(f"{_tmp}/losses.csv"))
        self.assertTrue(os.path.isdir(f"{_tmp}/samples"))
        self.assertTrue(os.path.isdir(f"{_tmp}/weights"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/0_100_x_gen.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/1_100_mask.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/2_100_foreground.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/3_100_background.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/weights/cgn_100.pth"))
        self.assertTrue(os.path.isfile(f"{_tmp}/weights/discriminator_100.pth"))
        self.assertTrue(best_epoch == 3)
        shutil.rmtree(_tmp)

    def test_fit_validation(self):
        model = CGNAnomaly(batch_size=32)
        X = X_raw[y_raw==0][:5000]
        X_val = X_raw[:5000]
        y_val = y_raw[:5000]
        _tmp = "./_cgn_anomaly_tmp"
        losses_all, (best_model, best_epoch) = model.fit(
            X, 
            X_val = X_val,
            y_val = y_val,
            n_epochs=30, 
            save_iter=500, 
            verb=True, 
            save_results=True, 
            save_path=_tmp, 
            workers=12
        )
        if best_epoch != 15:
            p = float(next(model.cgn.parameters()).data)
            best_p = float(next(best_model.cgn.parameters()).data)
            self.assertTrue(p != best_p)
            auc_model = compute_auc(y_val, model.predict(X_val))
            model.move_to('cpu')
            best_model.move_to('cuda')
            auc_best_model = compute_auc(y_val, best_model.predict(X_val))
            #self.assertTrue(auc_best_model >= auc_model)
        shutil.rmtree(_tmp)

    def test_fit_bw(self):
        model = CGNAnomaly(batch_size=32, img_channels=1)
        X = X_raw[y_raw==0][:5000][:,:1,:,:]
        _tmp = "./_cgn_anomaly_tmp"
        losses_all, (best_model, best_epoch) = model.fit(
            X, 
            n_epochs=1, 
            save_iter=100, 
            verb=True, 
            save_results=True, 
            save_path=_tmp, 
            workers=12
        )
        self.assertTrue(os.path.isfile(f"{_tmp}/cfg.yaml"))
        self.assertTrue(os.path.isfile(f"{_tmp}/losses.csv"))
        self.assertTrue(os.path.isdir(f"{_tmp}/samples"))
        self.assertTrue(os.path.isdir(f"{_tmp}/weights"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/0_100_x_gen.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/1_100_mask.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/2_100_foreground.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/3_100_background.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/weights/cgn_100.pth"))
        self.assertTrue(os.path.isfile(f"{_tmp}/weights/discriminator_100.pth"))
        shutil.rmtree(_tmp)

    def test_fit_bw_conv_disc(self):
        model = CGNAnomaly(batch_size=32, img_channels=1, disc_model="conv")
        X = X_raw[y_raw==0][:5000][:,:1,:,:]
        _tmp = "./_cgn_anomaly_tmp"
        losses_all, (best_model, best_epoch) = model.fit(
            X, 
            n_epochs=1, 
            save_iter=100, 
            verb=True, 
            save_results=True, 
            save_path=_tmp, 
            workers=12
        )
        self.assertTrue(os.path.isfile(f"{_tmp}/cfg.yaml"))
        self.assertTrue(os.path.isfile(f"{_tmp}/losses.csv"))
        self.assertTrue(os.path.isdir(f"{_tmp}/samples"))
        self.assertTrue(os.path.isdir(f"{_tmp}/weights"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/0_100_x_gen.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/1_100_mask.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/2_100_foreground.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/3_100_background.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/weights/cgn_100.pth"))
        self.assertTrue(os.path.isfile(f"{_tmp}/weights/discriminator_100.pth"))
        shutil.rmtree(_tmp)

# train a model for score testing
model = CGNAnomaly(batch_size=32)
X = X_raw[y_raw==0][:5000]
losses_all, best_model = model.fit(X, n_epochs=5, verb=True, save_results=False)
X_test = X_raw[y_raw!=0][:5000]

class TestPredict(unittest.TestCase):
    def compare_scores(self, score_type):
        scores = model.predict(X, score_type=score_type)
        scores_test = model.predict(X_test, score_type=score_type)
        self.assertTrue(len(scores)==5000)
        self.assertTrue(type(scores)==np.ndarray)
        self.assertTrue(scores.mean() < scores_test.mean())

    def test_disc_score(self):
        self.compare_scores("discriminator")

    def test_perc_score(self):
        self.compare_scores("perceptual")
