import torch
from torch import nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import copy
from tqdm import tqdm
from pathlib import Path
import time
import random
import pandas
import os

from sgad.utils import Optimizers, Subset
from sgad.sgvae import VAE, feature_matching_loss
from sgad.utils import save_cfg, Optimizers, compute_auc, Patch2Image, RandomCrop
from sgad.sgvae.utils import rp_trick, batched_score, logpx, get_float, Mean, logreg_fit, logreg_prob
from sgad.sgvae.utils import Discriminator, get_float
from sgad.shared.losses import BinaryLoss, MaskLoss, PerceptualLoss, PercLossText
from sgad.cgn.models.cgn import Reshape, init_net

def vaegan_generator_loss(sg, sr):
    """
    vaegan_generator_loss(scores_generated)
    
    mean(log(sg) + log(sr)) = E[log(D(G(z))) + log(D(G(E(x))))]
    """
    return - torch.sum(torch.log(sg + 1e-8) + torch.log(sr + 1e-8)) / 2

def vaegan_discriminator_loss(st, sg, sr):
    """
    vaegan_discriminator_loss(scores_true, scores_generated)
    
    mean(log(st) + log(1-sg) + log(1-sr))  = E[log(D(x)) + (log(1-D(G(z)))) + (log(1-D(G(E(x)))))]
    """
    return - torch.sum(torch.log(st + 1e-8) + (torch.log(1 - sg + 1e-8) + torch.log(1 - sr + 1e-8)) / 2)

class VAEGAN(nn.Module):
    """VAEGAN(**kwargs)
    
    kwargs = 
        fm_alpha=0.0,
        fm_depth=7,
        z_dim=32, 
        h_channels=32, 
        img_dim=32, 
        img_channels=3,
        init_type='orthogonal', 
        init_gain=0.1, 
        init_seed=None,
        batch_size=32, 
        vae_type="texture", 
        std_approx="exp",
        lr=0.0002,
        betas=[0.5, 0.999],
        device=None,
        log_var_x_estimate = "conv_net"
    """
    def __init__(self, 
            fm_alpha=1.0,
            fm_depth=7,
            input_range=[0,1],
            **kwargs):
        # supertype init
        super(VAEGAN, self).__init__()
                
        # vaes
        self.vae = VAE(**kwargs)
        
        # config
        self.config = copy.deepcopy(self.vae.config)
        self.config.fm_alpha = fm_alpha
        self.config.fm_depth = fm_depth
        self.config.input_range = input_range
        self.input_range = input_range
        self.device = self.vae.device
        self.z_dim = self.config.z_dim
        
        # seed
        init_seed = self.config.init_seed
        if init_seed is not None:
            torch.random.manual_seed(init_seed)

        # discriminator
        self.discriminator = Discriminator(
            self.config.img_channels, 
            self.config.h_channels, 
            self.config.img_dim
        )
        
        # modules for param groups
        self.params = nn.ModuleDict({
               "encoder" : nn.ModuleList([
                   self.vae.encoder,
                   self.vae.mu_net_z,
                   self.vae.log_var_net_z
               ]),
               "decoder" : nn.ModuleList([
                   self.vae.decoder,
                   self.vae.mu_net_x,
                   self.vae.log_var_net_x
               ]),
                "discriminator": self.discriminator
        })
        
        # optimizer
        self.opts = Optimizers()
        self.opts.set('encoder', self.params.encoder, lr=self.config.lr, betas=self.config.betas)
        self.opts.set('decoder', self.params.decoder, lr=self.config.lr, betas=self.config.betas)
        self.opts.set('discriminator', self.params.discriminator, lr=self.config.lr, betas=self.config.betas)        
        
        # reset seed
        torch.random.seed()

        # move to device
        self.move_to(self.device)

    def fit(self, X,
            n_epochs=100, 
            save_iter=1000, 
            verb=True, 
            save_weights=False,
            save_path=None, 
            workers=1,
            max_train_time=np.inf # in seconds           
           ):

        """Fit the model given data X.

        Returns (losses_all, best_model, best_epoch)
        """
        # first check if the data is as expected
        if X.max() != self.input_range[1] or X.min() != self.input_range[0]:
            raise ValueError(f'Expected data in range {self.input_range}, obtained [{X.min()}, {X.max()}]')

        # setup the dataloader
        y = torch.zeros(X.shape[0]).long()
        tr_loader = DataLoader(Subset(torch.tensor(X).float(), y), 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=workers)

        # setup save paths
        if save_path is not None:
            save_results = True
            model_path, sample_path, weights_path = self.setup_paths(
                save_path, save_weights, n_epochs, save_iter, workers)
            # samples for reconstruction
            x_sample = X[random.sample(range(X.shape[0]), 12),:,:,:]
        else:
            save_results = False

        # loss logging
        losses_all = {'iter': [], 'epoch': [], 'encl': [], 'decl': [], 'discl': [], 'kld': [], 'genl': [],
             'fml': [], 'xmax': []}

        self.train()
        pbar = tqdm(range(n_epochs))
        niter = 0
        start_time = time.time()
        for epoch in pbar:
            for i, data in enumerate(tr_loader):
                x = data['ims'].to(self.device)
                loss_vals = self.update_step(x)

                # log losses
                niter += 1
                self.log_losses(losses_all, niter, epoch, *loss_vals)

                # output
                if verb:                
                    self.print_output(pbar, i, tr_loader, *loss_vals)

                # saving weights and sample images
                batches_done = epoch * len(tr_loader) + i
                if save_results:
                    if batches_done % save_iter == 0:
                        print(f"Saving samples and weights to {model_path}")
                        self.save_sample_images(x_sample, sample_path, batches_done, n_cols=3)
                        outdf = pandas.DataFrame.from_dict(losses_all)
                        outdf.to_csv(os.path.join(model_path, "losses.csv"), index=False)
                        if save_weights:
                           self.save_weights(f"{weights_path}/sgvae_{batches_done:d}.pth")

                # exit if running for too long
                run_time = time.time() - start_time
                if run_time > max_train_time:
                    break

            # exit if running for too long
            if run_time > max_train_time:
                print("Given runtime exceeded, ending training prematurely.")
                break

        return losses_all

    def update_step(self, x):
        ### encoder block ###
        # kld
        mu_z, log_var_z = self.vae.encode(x)
        std_z = self.vae.std(log_var_z)
        z = rp_trick(mu_z, std_z)
        kld = torch.mean(self.vae.kld(mu_z, log_var_z))
        # fm loss
        x_rec = self.decode(z)
        fml = torch.mean(feature_matching_loss(x, x_rec, self.discriminator, self.config.fm_depth))
        # enc update
        self.opts.zero_grad(['encoder'])
        el = kld  + self.config.fm_alpha*fml
        el.backward(retain_graph=True)
        self.opts.step(['encoder'])

        ### decoder block ###
        # fm loss
        x_rec = self.decode(z.detach())
        fml = torch.mean(feature_matching_loss(x, x_rec, self.discriminator, self.config.fm_depth))
        # gen loss
        x_gen = self.generate(x.shape[0])
        sr = self.discriminator(x_rec)
        sg = self.discriminator(x_gen)
        gl = vaegan_generator_loss(sg, sr)
        # dec update
        self.opts.zero_grad(['decoder'])
        decl = self.config.fm_alpha*fml + gl
        decl.backward(retain_graph=True)
        self.opts.step(['decoder'])

        ### discriminator block ###
        # disc loss
        st = self.discriminator(x)
        srd = self.discriminator(x_rec.detach())
        sgd = self.discriminator(x_gen.detach())
        dl = vaegan_discriminator_loss(st, sgd, srd)
        # disc update
        self.opts.zero_grad(['discriminator'])
        dl.backward(retain_graph=False)
        self.opts.step(['discriminator'])

        # control how often the data is out of range
        x_rec_control, _ = self.vae.decode(z.detach())
        xmax = x_rec_control.max()

        return el, decl, dl, kld, gl, fml, xmax

    def log_losses(self, losses_all, niter, epoch, el, decl, dl, kld, gl, fml, xmax):
        losses_all['iter'].append(niter)
        losses_all['epoch'].append(epoch)
        losses_all['encl'].append(get_float(el))
        losses_all['decl'].append(get_float(decl))
        losses_all['discl'].append(get_float(dl))
        losses_all['kld'].append(get_float(kld))
        losses_all['genl'].append(get_float(gl))
        losses_all['fml'].append(get_float(fml))
        losses_all['xmax'].append(get_float(xmax))

    def print_output(self, pbar, i, tr_loader, el, decl, dl, kld, gl, fml, xmax):
        msg = f"[Batch {i}/{len(tr_loader)}]"
        msg += ''.join(f"[enc: {get_float(el):.3f}]")
        msg += ''.join(f"[dec: {get_float(decl):.3f}]")
        msg += ''.join(f"[disc: {get_float(dl):.3f}]")
        msg += ''.join(f"[kld: {get_float(kld):.3f}]")
        msg += ''.join(f"[gen: {get_float(gl):.3f}]")
        msg += ''.join(f"[fml: {get_float(fml):.3f}]")
        pbar.set_description(msg)

    def reconstruct(self, x):
        return torch.clamp(self.vae.reconstruct_mean(x), self.input_range[0], self.input_range[1])

    def decode(self, z):
        return torch.clamp(self.vae.decode(z)[0], self.input_range[0], self.input_range[1])

    def generate(self, x):
        return torch.clamp(self.vae.generate_mean(x), self.input_range[0], self.input_range[1])

    def discriminate(self, x):
        return self.discriminator(x).reshape(-1).detach().cpu().numpy()

    def discriminator_score(self, X, workers=1, batch_size=None, **kwargs):
        loader = self._create_score_loader(X, batch_size=batch_size, workers=workers)
        return batched_score(lambda x: 1 - self.discriminate(x), loader, self.device, **kwargs)
    
    def reconstruction_error(self, x, n=1):
        scores = []
        for i in range(n):
            rx = self.vae.reconstruct_mean(x)
            scores.append(nn.MSELoss(reduction='none')(rx, x).sum((1,2,3)).detach().cpu().numpy())

        return np.mean(scores, 0) 
        
    def reconstruction_score(self, X, workers=1, batch_size=None, **kwargs):
        loader = self._create_score_loader(X, batch_size=batch_size, workers=workers)
        return batched_score(self.reconstruction_error, loader, self.device, **kwargs)
    
    def fm_score(self, X, n=1, fm_depth=None)
        if fm_depth is None:
            fm_depth = self.config.fm_depth
        scores = []
        for i in range(n):
            fml = feature_matching_loss(x, self.vae.reconstruct_mean(x), self.discriminator, fm_depth)
            scores.append(fml.detach().cpu().numpy())

        return np.mean(scores, 0)

    def feature_matching_score(self, X, fm_depth=None, workers=1, batch_size=None, **kwargs):
        loader = self._create_score_loader(X, batch_size=batch_size, workers=workers)
        return batched_score(self._fm_score, loader, self.device, **kwargs)
    
    def predict(self, X, score_type="discriminator", **kwargs):
        if score_type == "discriminator":
            return self.discriminator_score(X, **kwargs)
        elif score_type == "feature_matching":
            return self.feature_matching_score(X, **kwargs)
        elif score_type == "reconstruction":
            return self.reconstruction_score(X, **kwargs)
        else:
            raise ValueError(f"Unknown score type {score_type}")

    def num_params(self):
        s = 0
        for p in self.parameters():
            s += np.array(p.data.to('cpu')).size
        return s

    def save_weights(self, file):
        torch.save(self.state_dict(), file)

    def save_sample_images(self, x, sample_path, batches_done, n_cols=3):
        """Saves a grid of generated and reconstructed digits"""
        x_gen = [self.generate(10).to('cpu') for _ in range(n_cols)]
        x_gen = torch.concat(x_gen, 0)
        _x = torch.tensor(x).to(self.device).float()
        x_reconstructed = self.reconstruct(_x).to('cpu')
        _x = _x.to('cpu')

        def save(x, path, n_cols, sz=64):
            x = F.interpolate(x, (sz, sz))
            save_image(x.data, path, nrow=n_cols, normalize=True, padding=2)

        Path(sample_path).mkdir(parents=True, exist_ok=True)
        save(x_gen.data, f"{sample_path}/0_{batches_done:d}_x_gen.png", n_cols, sz=self.config.img_dim)
        save(torch.concat((_x, x_reconstructed), 0).data, f"{sample_path}/1_{batches_done:d}_x_reconstructed.png", n_cols*2, sz=self.config.img_dim)
        
    def setup_paths(self, save_path, save_weights, n_epochs, save_iter, workers, cfg=None):
        print(f'Creating save path {save_path}.')
        model_path = Path(save_path)
        weights_path = model_path / 'weights'
        sample_path = model_path / 'samples'
        if save_weights:
            weights_path.mkdir(parents=True, exist_ok=True)
        else:
            print("If you want to save weights, set save_weights=True.")
        sample_path.mkdir(parents=True, exist_ok=True)

        # dump config
        if cfg is None:
            cfg = self.config
        cfg = copy.deepcopy(cfg)
        cfg.n_epochs = n_epochs
        cfg.save_iter = save_iter
        cfg.save_path = save_path
        cfg.workers = workers
        save_cfg(cfg, model_path / "cfg.yaml")

        return model_path, sample_path, weights_path

    def move_to(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        # push to device and train
        self.device = device
        self.vae.move_to(device)
        self.discriminator = self.discriminator.to(device)
        self = self.to(device)
        
    def _create_score_loader(self, X, batch_size=None, workers=1):
        # create the loader
        if batch_size is None:
            batch_size = self.config.batch_size
        y = torch.zeros(X.shape[0]).long()
        loader = DataLoader(Subset(torch.tensor(X).float(), y), 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=workers)
        return loader
    