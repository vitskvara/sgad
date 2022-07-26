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
from itertools import chain

from sgad.utils import Optimizers, Subset
from sgad.sgvae import SGVAE, feature_matching_loss
from sgad.utils import save_cfg, Optimizers, compute_auc, Patch2Image, RandomCrop
from sgad.sgvae.utils import rp_trick, batched_score, logpx, get_float, Mean, logreg_fit, logreg_prob
from sgad.sgvae.utils import Discriminator, create_score_loader
from sgad.shared.losses import BinaryLoss, MaskLoss, PerceptualLoss, PercLossText
from sgad.cgn.models.cgn import Reshape, init_net
from sgad.sgvae.vaegan import vaegan_generator_loss, vaegan_discriminator_loss

class SGVAEGAN(nn.Module):
    """SGVAEGAN(**kwargs)
    
    kwargs = 
        fm_alpha=0.0,
        fm_depth=7,
        input_range=[0,1],
        z_dim=32, 
        h_channels=32, 
        img_dim=32, 
        img_channels=3,
        weights_texture = [0.01, 0.05, 0.0, 0.01], 
        weight_binary=1.0,
        weight_mask=1.0,
        tau_mask=0.1,       
        log_var_x_estimate_top = "global",
        alpha = None,
        latent_structure="independent",
        fixed_mask_epochs=1,
        init_type='orthogonal', 
        init_gain=0.1, 
        init_seed=None,
        batch_size=1, 
        std_approx="exp",
        lr=0.0002,
        betas=[0.5, 0.999],
        device=None
    """
    def __init__(self, 
            fm_alpha=1.0,
            fm_depth=7,
            input_range=[0,1],
            alpha=None,
            **kwargs):
        # supertype init
        super(SGVAEGAN, self).__init__()
                
        # vaes
        self.sgvae = SGVAE(**kwargs)
        
        # config
        self.config = copy.deepcopy(self.sgvae.config)
        self.config.fm_alpha = fm_alpha
        self.config.fm_depth = fm_depth
        self.device = self.sgvae.device
        self.z_dim = self.config.z_dim
        self.config.input_range = input_range
        self.input_range = input_range

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
        
        # parameter groups
        # we optimize encoders, decoders and the discriminator individually
        self.params = nn.ModuleDict({
            'encoders': nn.ModuleList([
                self.sgvae.vae_shape.encoder,
                self.sgvae.vae_shape.mu_net_z,
                self.sgvae.vae_shape.log_var_net_z,
                self.sgvae.vae_background.encoder,
                self.sgvae.vae_background.mu_net_z,
                self.sgvae.vae_background.log_var_net_z,
                self.sgvae.vae_foreground.encoder,
                self.sgvae.vae_foreground.mu_net_z,
                self.sgvae.vae_foreground.log_var_net_z
            ]),
            'decoders': nn.ModuleList([
                self.sgvae.vae_shape.decoder,
                self.sgvae.vae_shape.mu_net_x,
                self.sgvae.vae_background.decoder,
                self.sgvae.vae_background.mu_net_x,
                self.sgvae.vae_foreground.decoder,
                self.sgvae.vae_foreground.mu_net_x,
            ]),
            'discriminator': nn.ModuleList([self.discriminator]),
        })
        
        # optimizer
        self.opts = Optimizers()
        self.opts.set('encoders', self.params.encoders, lr=self.config.lr, betas=self.config.betas)
        self.opts.set('decoders', self.params.decoders, lr=self.config.lr, betas=self.config.betas)
        self.opts.set('discriminator', self.params.discriminator, lr=self.config.lr, 
                      betas=self.config.betas)        
        
        # alphas for joint prediction
        self.set_alpha(alpha, alpha_score_type=None)

        # reset seed
        torch.random.seed()

        # move to device
        self.move_to(self.device)

    def fit(self, X,
            n_epochs=20, 
            save_iter=1000, 
            verb=True, 
            save_weights=False,
            save_path=None, 
            workers=12,
            max_train_time=np.inf # in seconds           
           ):
        """Fit the model given X.

        Returns (losses_all, None, None).
        """
        # setup the dataloader
        y = torch.zeros(X.shape[0]).long() if y is None else y
        tr_loader = DataLoader(Subset(torch.tensor(X).float(), y), 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=workers)
        
        # loss logging
        losses_all = {'iter': [], 'epoch': [], 'encl': [], 'decl': [], 'discl': [], 'kld': [], 'genl': [],
             'fml': [], 'mask': [], 'texture': [], 'binary': [],
             'kld_shape': [], 'kld_background': [], 'kld_foreground': []}

        # setup save paths
        if save_path is not None:
            save_results = True
            model_path, sample_path, weights_path = self.sgvae.setup_paths(
                save_path, save_weights, n_epochs, save_iter, workers, cfg=self.config)
            # samples for reconstruction
            x_sample = X[random.sample(range(X.shape[0]), 12),:,:,:]
        else:
            save_results = False

        # tracking
        self.train()
        pbar = tqdm(range(n_epochs))
        niter = 0
        start_time = time.time()
        for iepoch, epoch in enumerate(pbar):
            for i, batch in enumerate(tr_loader):
                x = batch['ims'].to(self.device)
        
                # do the updates
                z_s,z_b,z_f,x_rec,kld,bin_l,mask_l,text_l,fml,el,kld_s,kld_b,kld_f= self.update_encoders(x, 
                    iepoch)
                x_rec,x_gen,bin_l,mask_l,text_l,fml,gl,decl = self.update_decoders(x, z_s, z_b, z_f, 
                    iepoch)
                dl = self.update_discriminator(x, x_rec, x_gen)
                                
                # collect losses
                niter += 1
                self.log_losses(losses_all, niter, epoch, el, decl, dl, kld, gl, fml, mask_l, text_l, 
                    bin_l, kld_s, kld_b, kld_f)

                # output
                if verb:
                    self.print_progress(pbar, i, len(tr_loader), el, decl, dl, kld, gl, fml, bin_l, 
                        mask_l, text_l)

                # Saving
                batches_done = epoch * len(tr_loader) + i
                if save_results:
                    if batches_done % save_iter == 0:
                        print(f"Saving samples and weights to {model_path}")
                        self.sgvae.save_sample_images(x_sample, sample_path, batches_done, n_cols=3)
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

        return losses_all, None, None

    def _common_losses(self, x, z_s, z_b, z_f, iepoch):
        """This is used to compute loss values and other stuff for updates of the encoders and decoders."""
        # get the reconstructed image
        if iepoch >= self.sgvae.fixed_mask_epochs:    
            mask = self.sgvae._decode(self.sgvae.vae_shape, z_s)
            mask = torch.clamp(mask, 0.0001, 0.9999).repeat(1, self.config.img_channels, 1, 1)
        else:
            mask = self.sgvae.fixed_mask(x,r=0.2)
        background = self.sgvae._decode(self.sgvae.vae_background, z_b)
        foreground = self.sgvae._decode(self.sgvae.vae_foreground, z_f)
        x_rec = self.compose_image(mask, background, foreground)
        
        # get binary loss
        bin_l = self.sgvae.binary_loss(mask)
        mask_l = self.sgvae.mask_loss(mask).mean()

        # get the texture loss
        text_l = self.sgvae.texture_loss(x, mask, foreground)

        # fm loss
        fml = torch.mean(feature_matching_loss(x, x_rec, self.discriminator, self.config.fm_depth))
        
        return bin_l, mask_l, text_l, fml, x_rec

    def update_encoders(self, x, iepoch):
        """One optimization step of the encoders"""
        # get the klds and zs
        z_s, kld_s = self.sgvae._encode(self.sgvae.vae_shape, x)
        z_b, kld_b = self.sgvae._encode(self.sgvae.vae_background, x)
        z_f, kld_f = self.sgvae._encode(self.sgvae.vae_foreground, x)
        kld = torch.mean(kld_s) + torch.mean(kld_b) + torch.mean(kld_f)

        # get the common stuff - unfortunatelly I did not figure out how to only compute this once
        # both for encoder and decoder pass
        bin_l, mask_l, text_l, fml, x_rec = self._common_losses(x, z_s, z_b, z_f, iepoch)

        # param update
        self.opts.zero_grad(['encoders'])
        el = kld + self.config.fm_alpha*fml + bin_l + mask_l + text_l
        el.backward()
        self.opts.step(['encoders'], False)
        
        return z_s, z_b, z_f, x_rec, kld, bin_l, mask_l, text_l, fml, el, kld_s, kld_b, kld_f

    def update_decoders(self, x, z_s, z_b, z_f, iepoch):
        """One optimization step of the decoders"""
        # detach the zs - we update the encoder in a different place
        z_s = z_s.detach()
        z_b = z_b.detach()
        z_f = z_f.detach()
        
        # get the common stuff
        bin_l, mask_l, text_l, fml, x_rec = self._common_losses(x, z_s, z_b, z_f, iepoch)

        # also create a generated sample
        x_gen = self.generate(x.shape[0])
        
        # generator loss
        sr = self.discriminator(x_rec)
        sg = self.discriminator(x_gen)
        gl = vaegan_generator_loss(sg, sr)
        
        # param update
        self.opts.zero_grad(['decoders'])
        dl = self.config.fm_alpha*fml + gl + bin_l + mask_l + text_l
        dl.backward()
        self.opts.step(['decoders'], False)
        
        return x_rec, x_gen, bin_l, mask_l, text_l, fml, gl, dl
        
    def update_discriminator(self, x, x_rec, x_gen):
        """One optimization step of the discriminator"""
        # discriminator loss
        st = self.discriminator(x)
        srd = self.discriminator(x_rec.detach())
        sgd = self.discriminator(x_gen.detach())
        discl = vaegan_discriminator_loss(st, sgd, srd)
        
        # param update
        self.opts.zero_grad(['discriminator'])
        discl.backward()
        self.opts.step(['discriminator'], False)
        
        return discl

    def log_losses(self, losses_all, niter, epoch, el, decl, dl, kld, gl, fml, mask_l, text_l, bin_l, 
                     kld_s, kld_b, kld_f):
        losses_all['iter'].append(niter)
        losses_all['epoch'].append(epoch)
        losses_all['encl'].append(get_float(el))
        losses_all['decl'].append(get_float(decl))
        losses_all['discl'].append(get_float(dl))
        losses_all['kld'].append(get_float(kld))
        losses_all['genl'].append(get_float(gl))
        losses_all['fml'].append(get_float(fml))
        losses_all['mask'].append(get_float(mask_l))
        losses_all['texture'].append(get_float(text_l))
        losses_all['binary'].append(get_float(bin_l))
        losses_all['kld_shape'].append(get_float(kld_s))
        losses_all['kld_background'].append(get_float(kld_b))
        losses_all['kld_foreground'].append(get_float(kld_f))

    def print_progress(self, pbar, i, nsteps, el, decl, dl, kld, gl, fml, bin_l, mask_l, text_l, *args):
        msg = f"[Batch {i}/{nsteps}]"
        msg += ''.join(f"[enc: {get_val(el):.3f}]")
        msg += ''.join(f"[dec: {get_val(decl):.3f}]")
        msg += ''.join(f"[disc: {get_val(dl):.3f}]")
        msg += ''.join(f"[kld: {get_val(kld):.3f}]")
        msg += ''.join(f"[gen: {get_val(gl):.3f}]")
        msg += ''.join(f"[fml: {get_val(fml):.3f}]")
        msg += ''.join(f"[binary: {get_float(bin_l):.3f}]")
        msg += ''.join(f"[mask: {get_float(mask_l):.3f}]")
        msg += ''.join(f"[texture: {get_float(text_l):.3f}]")
        pbar.set_description(msg)

    def save_weights(self, file):
        torch.save(self.state_dict(), file)

    def clamp(self, x):
        return torch.clamp(x, self.input_range[0], self.input_range[1])
        
    def reconstruct(self, x):
        return self.clamp(self.sgvae.reconstruct_mean(x))
    
    def generate(self, n):
        return self.clamp(self.sgvae.generate_mean(n))

    def compose_image(self, mask, background, foreground):
        return self.clamp(self.sgvae.compose_image(mask, background, foreground))
    
    def mask(self, x):
        """Extract the mask of x."""
        return self.sgvae.mask(x)

    def background(self, x):
        """Extract the background of x."""
        return self.sgvae.background(x)
    
    def foreground(self, x):
        """Extract the foreground of x."""
        return self.sgvae.foreground(x)
    
    def discriminate(self, x):
        return self.discriminator(x).reshape(-1).detach().cpu().numpy()

    def discriminator_score(self, X, workers=1, batch_size=None, **kwargs):
        loader = create_score_loader(X, batch_size if batch_size is not None else self.batch_size, 
            workers=workers, shuffle=False)
        return batched_score(lambda x: 1 - self.discriminate(x), loader, self.device, **kwargs)
    
    def reconstruction_error(self, x, n=1):
        scores = []
        for i in range(n):
            rx = self.reconstruct(x)
            scores.append(nn.MSELoss(reduction='none')(rx, x).sum((1,2,3)).detach().cpu().numpy())

        return np.mean(scores, 0) 
        
    def reconstruction_score(self, X, workers=1, batch_size=None, **kwargs):
        loader = create_score_loader(X, batch_size if batch_size is not None else self.batch_size, 
            workers=workers, shuffle=False)
        return batched_score(self.reconstruction_error, loader, self.device, **kwargs)
    
    def fm_score(self, x, n=1, fm_depth=None):
        if fm_depth is None:
            fm_depth = self.config.fm_depth
        scores = []
        for i in range(n):
            fml = feature_matching_loss(x, self.reconstruct(x), self.discriminator, fm_depth)
            scores.append(fml.detach().cpu().numpy())

        return np.mean(scores, 0)

    def feature_matching_score(self, X, fm_depth=None, workers=1, batch_size=None, **kwargs):
        loader = create_score_loader(X, batch_size if batch_size is not None else self.batch_size, 
            workers=workers, shuffle=False)
        return batched_score(self.fm_score, loader, self.device, **kwargs)
    
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

    def move_to(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        # push to device and train
        self.device = device
        self.sgvae.move_to(device)
        self.discriminator = self.discriminator.to(device)
        self = self.to(device)

    def set_alpha(self, alpha, alpha_score_type):
        if alpha is None:
            self.alpha = None
            self.alpha_score_type = None
            return

        if not len(alpha) == 6:
            raise ValueError("given alpha must be a vector of length 6")
        self.alpha = np.array(alpha).reshape(6,)

