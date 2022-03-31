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
from sklearn.linear_model import LogisticRegression
import warnings

from sgad.utils import Optimizers, Subset
from sgad.sgvae import VAE
from sgad.utils import save_cfg, Optimizers, compute_auc, Patch2Image, RandomCrop
from sgad.sgvae.utils import rp_trick, batched_score, logpx, get_float, Mean
from sgad.shared.losses import BinaryLoss, MaskLoss, PerceptualLoss, PercLossText
from sgad.cgn.models.cgn import Reshape, init_net

# Shape-Guided Variational AutoEncoder
class SGVAE(nn.Module):
    """SGVAE(**kwargs)
    
    kwargs = 
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
        fixed_mask_epochs=0,
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
            weights_texture = [0.01, 0.05, 0.0, 0.01], 
            weight_binary=1.0,
            weight_mask=1.0, 
            tau_mask=0.1, 
            std_approx="exp", 
            device=None, 
            latent_structure="independent", 
            fixed_mask_epochs=0,
            log_var_x_estimate_top = "global",
            alpha = None, # or any numpy vector
            **kwargs):
        # supertype init
        super(SGVAE, self).__init__()
                
        # vaes
        self.vae_shape = VAE(vae_type="shape", std_approx=std_approx, **kwargs)
        self.vae_background = VAE(vae_type="texture", std_approx=std_approx, **kwargs)
        self.vae_foreground = VAE(vae_type="texture", std_approx=std_approx, **kwargs)
        
        # config
        self.config = copy.deepcopy(self.vae_background.config)
        self.device = self.vae_shape.device
        self.config.weight_binary = weight_binary
        self.config.weight_mask = weight_mask
        self.config.tau_mask = tau_mask
        self.z_dim = self.config.z_dim
        self.config.pop('vae_type')

        # seed
        init_seed = self.config.init_seed
        if init_seed is not None:
            torch.random.manual_seed(init_seed)

        # mask and texture loss
        self.binary_loss = BinaryLoss(weight_binary)
        self.mask_loss = MaskLoss(weight_mask, interval=[tau_mask, 1-tau_mask])
        self.config.weights_texture = weights_texture
        self.texture_loss = PercLossText(weights_texture, patch_sz=[8, 8], im_sz=self.config.img_dim, 
            sample_sz=100, n_up=6)

        # training type
        possible_structures = ["independent", "mask_dependent"]
        if not latent_structure in possible_structures:
            raise ValueError(f'Required latent structure {latent_structure} unknown. Choose one of {possible_structures}.')
        self.latent_structure = latent_structure
        self.config.latent_structure = latent_structure
        self.fixed_mask_epochs = fixed_mask_epochs
        self.config.fixed_mask_epochs = fixed_mask_epochs

        # x log var
        self.config.log_var_x_estimate_top = log_var_x_estimate_top
        if log_var_x_estimate_top == "conv_net":
            self.log_var_net_x = nn.Sequential(
                nn.Conv2d(self.vae_foreground.out_channels, 1, 3, stride=1, padding=1, bias=False),
                Mean(1,2,3),
                Reshape(-1,1,1,1)
                )
            self.log_var_x_global = None
            init_net(self.log_var_net_x, init_type=self.config.init_type, init_gain=self.config.init_gain)
        elif log_var_x_estimate_top == "global":
            self.log_var_x_global = nn.Parameter(torch.Tensor([-1.0])) # nn.Parameter(torch.Tensor([0.0]))
            self.log_var_net_x = lambda x: self.log_var_x_global
        else:
            warnings.warn(f"log_var_x_estimate_top {log_var_x_estimate_top} not known, you should set .log_var_net_x with a callable function")

        # optimizer
        self.opts = Optimizers()
        self.opts.set('sgvae', self, lr=self.config.lr, betas=self.config.betas)        
        
        # alphas for joint prediction
        self.set_alpha(alpha)

        # reset seed
        torch.random.seed()

        # move to device
        self.move_to(device)

    def fit(self, X, y=None,
            X_val=None, y_val=None,
            n_epochs=20, 
            save_iter=1000, 
            verb=True, 
            save_weights=False,
            save_path=None, 
            workers=12,
            max_train_time=np.inf # in seconds           
           ):
        """Fit the model given X (and possibly y).

        If y is supported, the classes should be labeled by integers like in range(n_classes).

        Returns (losses_all, best_model, best_epoch)
        """
        # setup the dataloader
        y = torch.zeros(X.shape[0]).long() if y is None else y
        tr_loader = DataLoader(Subset(torch.tensor(X).float(), y), 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=workers)
        
        # also check the validation data
        if X_val is not None and y_val is None:
            raise ValueError("X_val given without y_val - please provide it as well.")
        auc_val = best_auc_val = -1.0
        
        # loss values, n_epochs, save_iter, workers
        losses_all = {'iter': [], 'epoch': [], 'loss': [], 'kld': [], 'logpx': [], 'elbo': [], 'auc_val': [],
                     'kld_shape': [], 'kld_background': [], 'kld_foreground': [], 'binary': [],
                     'mask': [], 'texture': []}

        # setup save paths
        if save_path is not None:
            save_results = True
            model_path, sample_path, weights_path = self.setup_paths(
                save_path, save_weights, n_epochs, save_iter, workers)
            # samples for reconstruction
            x_sample = X[random.sample(range(X.shape[0]), 12),:,:,:]
        else:
            save_results = False

        # tracking
        pbar = tqdm(range(n_epochs))
        niter = 0
        start_time = time.time()
        for iepoch, epoch in enumerate(pbar):
            for i, batch in enumerate(tr_loader):
                if self.latent_structure == "independent":
                    loss_values = self.train_step_independent(batch, iepoch)
                elif self.latent_structure == "mask_dependent":
                    loss_values = self.train_step_mask_dependent(batch, iepoch)
                
                # collect losses
                self.log_losses(losses_all, niter, epoch, auc_val, *loss_values)

                # output
                if verb:
                    self.print_progress(pbar, i, len(tr_loader), auc_val, *loss_values)

                # Saving
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

            # after every epoch, compute the val auc
            if X_val is not None:
                self.eval()
                scores_val = self.predict(X_val, score_type='logpx', num_workers=workers, n=10)
                auc_val = compute_auc(y_val, scores_val)
                # also copy the params
                if auc_val > best_auc_val:
                    best_model = self.cpu_copy()
                    best_epoch = epoch+1
                    best_auc_val = auc_val
                self.train()

            # exit if running for too long
            if run_time > max_train_time:
                print("Given runtime exceeded, ending training prematurely.")
                break

        # return self as best model
        if X_val is None:
            best_model = self.cpu_copy()
            best_epoch = n_epochs

        return losses_all, best_model, best_epoch

    def train_step_independent(self, batch, iepoch):
        # Data to device
        x = batch['ims'].to(self.device)
        
        # now get the klds and zs
        z_s, kld_s = self._encode(self.vae_shape, x)
        z_b, kld_b = self._encode(self.vae_background, x)
        z_f, kld_f = self._encode(self.vae_foreground, x)

        # now get the decoded outputs
        if iepoch >= self.fixed_mask_epochs:    
            mask = self._decode(self.vae_shape, z_s)
            mask = torch.clamp(mask, 0.0001, 0.9999).repeat(1, self.config.img_channels, 1, 1)
        else:
            mask = self.fixed_mask(x,r=0.2)
    
        background = self._decode(self.vae_background, z_b)
        foreground = self._decode(self.vae_foreground, z_f)
        
        # merge them together
        x_hat = self.compose_image(mask, background, foreground)
        mu_x = x_hat
        log_var_x = self.log_var_net_x(x_hat)
        std_x = self.std(log_var_x)
        lpx = torch.mean(logpx(x, mu_x, std_x))
        
        # get elbo
        kld_s = torch.mean(kld_s)
        kld_b = torch.mean(kld_b)
        kld_f = torch.mean(kld_f)
        kld = kld_s + kld_b + kld_f
        elbo = kld - lpx
        
        # get binary loss
        bin_l = self.binary_loss(mask)
        mask_l = self.mask_loss(mask).mean()

        # get the texture loss
        text_l = self.texture_loss(x, mask, foreground)

        # do a step    
        self.opts.zero_grad(['sgvae'])
        l = elbo + bin_l + mask_l + text_l
        l.backward()
        self.opts.step(['sgvae'], False)

        return l, elbo, kld, lpx, bin_l, mask_l, text_l, kld_s, kld_b, kld_f

    def train_step_mask_dependent(self, batch, iepoch):
        # Data to device
        x = batch['ims'].to(self.device)
        
        # first get the mask
        z_s, kld_s = self._encode(self.vae_shape, x)
        if iepoch >= self.fixed_mask_epochs:    
            mask = self._decode(self.vae_shape, z_s)
            mask = torch.clamp(mask, 0.0001, 0.9999).repeat(1, self.config.img_channels, 1, 1)
        else:
            mask = self.fixed_mask(x,r=0.2)
            
        # now mask the input
        x_m = x * mask.detach()
        
        # now use this as input to the remaining vaes
        z_b, kld_b = self._encode(self.vae_background, x_m)
        z_f, kld_f = self._encode(self.vae_foreground, x_m)
        background = self._decode(self.vae_background, z_b)
        foreground = self._decode(self.vae_foreground, z_f)
        
        # merge them together
        x_hat = self.compose_image(mask, background, foreground)

        # and reconstruct
        mu_x = x_hat
        log_var_x = self.log_var_net_x(x_hat)
        std_x = self.std(log_var_x)
        lpx = torch.mean(logpx(x, mu_x, std_x))
        
        # get elbo
        kld_s = torch.mean(kld_s)
        kld_b = torch.mean(kld_b)
        kld_f = torch.mean(kld_f)
        kld = kld_s + kld_b + kld_f
        elbo = kld - lpx
        
        # get mask loss
        bin_l = self.binary_loss(mask)
        mask_l = self.mask_loss(mask).mean()
        
        # get the texture loss
        text_l = self.texture_loss(x, mask, foreground)

        # do a step    
        self.opts.zero_grad(['sgvae'])
        l = elbo + bin_l + mask_l + text_l
        l.backward()
        self.opts.step(['sgvae'], False)

        return l, elbo, kld, lpx, bin_l, mask_l, text_l, kld_s, kld_b, kld_f

    def setup_paths(self, save_path, save_weights, n_epochs, save_iter, workers):
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
        cfg = copy.deepcopy(self.config)
        cfg.n_epochs = n_epochs
        cfg.save_iter = save_iter
        cfg.save_path = save_path
        cfg.workers = workers
        save_cfg(cfg, model_path / "cfg.yaml")

        return model_path, sample_path, weights_path
        
    def log_losses(self, losses_all, niter, epoch, auc_val, l, elbo, kld, lpx, bin_l, mask_l, text_l,
                     kld_s, kld_b, kld_f):
        niter += 1
        losses_all['iter'].append(niter)
        losses_all['epoch'].append(epoch)
        losses_all['loss'].append(get_float(l))
        losses_all['elbo'].append(get_float(elbo))
        losses_all['kld'].append(get_float(kld))
        losses_all['logpx'].append(get_float(lpx))
        losses_all['auc_val'].append(auc_val)
        losses_all['kld_shape'].append(get_float(kld_s))
        losses_all['kld_background'].append(get_float(kld_b))
        losses_all['kld_foreground'].append(get_float(kld_f))
        losses_all['binary'].append(get_float(bin_l))
        losses_all['mask'].append(get_float(mask_l))
        losses_all['texture'].append(get_float(text_l))

    def print_progress(self, pbar, i, nsteps, auc_val, l, elbo, kld, lpx, bin_l, mask_l, text_l, *args):
        msg = f"[Batch {i}/{nsteps}]"
        msg += ''.join(f"[loss: {get_float(l):.3f}]")
        msg += ''.join(f"[elbo: {get_float(elbo):.3f}]")
        msg += ''.join(f"[kld: {get_float(kld):.3f}]")
        msg += ''.join(f"[logpx: {get_float(lpx):.3f}]")
        msg += ''.join(f"[binary: {get_float(bin_l):.3f}]")
        msg += ''.join(f"[mask: {get_float(mask_l):.3f}]")
        msg += ''.join(f"[texture: {get_float(text_l):.3f}]")
        msg += ''.join(f"[auc val: {auc_val:.3f}]")
        pbar.set_description(msg)

    def fixed_mask(self, x, r=0.1):
        """Create a fixed mask where (1-r) ratio of inner pixels are ones."""
        mask = torch.zeros(x.shape)
        sz = np.array(mask.shape[2:])
        mask[:,:,round(sz[0]*r):round(sz[0]*(1-r)), round(sz[1]*r):round(sz[1]*(1-r))] = 1
        mask = mask.to(self.device)
        return mask

    def std(self, log_var):
        if self.config.std_approx == "exp":
            return torch.exp(log_var/2)
        else:
            return torch.nn.Softplus(log_var/2) + np.float32(1e-6)
       
    def _encode(self, vae, x):
        """Returns sampled z and kld for given vae."""
        mu_z, log_var_z = vae.encode(x)
        std_z = vae.std(log_var_z)
        z = rp_trick(mu_z, std_z)
        kld = vae.kld(mu_z, log_var_z)
        return z, kld
    
    def _decode(self, vae, z):
        """Returns mean decoded x for given vae."""
        mu_x, log_var_x = vae.decode(z)
        return mu_x
    
    def compose_image(self, mask, background, foreground):
        if self.latent_structure == "independent":
            return mask * foreground + (1 - mask) * background
        elif self.latent_structure == "mask_dependent":
            return mask * foreground + (1 - mask) * background #foreground + background

    def encode(self, x):
        """For given x, returns means and logvars in z space for all vaes in the model."""
        encodings_s = self.vae_shape.encode(x)
        encodings_b = self.vae_background.encode(x)
        encodings_f = self.vae_foreground.encode(x)
        return encodings_s, encodings_b, encodings_f
    
    def encoded(self, x):
        """For given x, returns samples in z space for all vaes in the model."""
        encodings_s = self.vae_shape.encoded(x)
        encodings_b = self.vae_background.encoded(x)
        encodings_f = self.vae_foreground.encoded(x)
        return encodings_s, encodings_b, encodings_f

    def decode(self, zs):
        """For given set of z space encodings, returns means and logvars for all vaes."""
        decodings_s = self.vae_shape.decode(zs[0])
        decodings_b = self.vae_background.decode(zs[1])
        decodings_f = self.vae_foreground.decode(zs[2])
        return decodings_s, decodings_b, decodings_f
    
    def decoded(self, zs):
        """For given set of z space encodings, returns samples in x space for all vaes. Don't use this for training!"""
        decodings_s = self.vae_shape.decoded(zs[0])
        decodings_b = self.vae_background.decoded(zs[1])
        decodings_f = self.vae_foreground.decoded(zs[2])
        return decodings_s, decodings_b, decodings_f

    def decode_image_components(self, zs):
        """Using the means in x space produced by vaes, return the individual image components."""
        (mask, _), (background, _), (foreground, _) = self.decode(zs)
        mask = torch.clamp(mask, 0.0001, 0.9999).repeat(1, self.config.img_channels, 1, 1)
        
        return mask, background, foreground

    def decode_image(self, zs):
        """For given set of z space encodings, returns a sampled final image."""
        mask, background, foreground = self.decode_image_components(zs)

        # now get he man and std and sample
        mu_x = self.compose_image(mask, background, foreground)
        
        log_var_x = self.log_var_net_x(mu_x)
        std_x = self.std(log_var_x)
        return rp_trick(mu_x, std_x)

    def decode_image_mean(self, zs):
        """For given set of z space encodings, returns a mean final image."""
        mask, background, foreground = self.decode_image_components(zs)
        return self.compose_image(mask, background, foreground)

    def forward(self, x):
        """Returns clamped mask, background, foreground."""
        if self.latent_structure == "independent":               
            return self.decode_image_components(self.encoded(x))
        elif self.latent_structure == "mask_dependent":
            # first get the mask
            z_s, kld_s = self._encode(self.vae_shape, x)
            mask = self._decode(self.vae_shape, z_s)
            mask = torch.clamp(mask, 0.0001, 0.9999).repeat(1, self.config.img_channels, 1, 1)
            
            # now mask the input
            x_m = x * mask        

            # now use this as input to the remaining vaes
            z_b, kld_b = self._encode(self.vae_background, x_m)
            z_f, kld_f = self._encode(self.vae_foreground, x_m)
            background = self._decode(self.vae_background, z_b)
            foreground = self._decode(self.vae_foreground, z_f)
                        
            return mask, background, foreground

    def reconstruct(self, x):
        """Returns sampled reconstruction of x."""
        mask, background, foreground = self(x)        
        mu_x = self.compose_image(mask, background, foreground)
        log_var_x = self.log_var_net_x(mu_x)
        std_x = self.std(log_var_x)
        return rp_trick(mu_x, std_x)

    def reconstruct_mean(self, x):
        """Returns mean reconstruction of x."""
        mask, background, foreground = self(x)        
        return self.compose_image(mask, background, foreground)

    def generate(self, n):
        p = torch.distributions.Normal(torch.zeros(n, self.z_dim), 1.0)
        zs = [p.sample().to(self.device) for _ in range(3)]
        return self.decode_image(zs)

    def generate_mean(self, n):
        p = torch.distributions.Normal(torch.zeros(n, self.z_dim), 1.0)
        zs = [p.sample().to(self.device) for _ in range(3)]
        return self.decode_image_mean(zs)
    
    def move_to(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        # push to device and train
        self.device = device
        self.vae_shape.move_to(device)
        self.vae_background.move_to(device)
        self.vae_foreground.move_to(device)
        self = self.to(device)

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
        x_gen_mean = [self.generate_mean(10).to('cpu') for _ in range(n_cols)]
        x_gen_mean = torch.concat(x_gen_mean, 0)
        _x = torch.tensor(x).to(self.device).float()
        mask, background, foreground = self(_x)
        x_reconstructed = self.reconstruct(_x)
        x_reconstructed = torch.concat((x_reconstructed, mask, foreground, background), 0)
        x_reconstructed = x_reconstructed.to('cpu')
        x_reconstructed_mean = self.reconstruct_mean(_x)
        x_reconstructed_mean = torch.concat((x_reconstructed_mean, mask, foreground, background), 0)
        x_reconstructed_mean = x_reconstructed_mean.to('cpu')
        _x = _x.to('cpu')

        def save(x, path, n_cols, sz=64):
            x = F.interpolate(x, (sz, sz))
            save_image(x.data, path, nrow=n_cols, normalize=True, padding=2)

        Path(sample_path).mkdir(parents=True, exist_ok=True)
        save(x_gen.data, f"{sample_path}/0_{batches_done:d}_x_gen.png", n_cols, sz=self.config.img_dim)
        save(x_gen_mean.data, f"{sample_path}/1_{batches_done:d}_x_gen_mean.png", n_cols, sz=self.config.img_dim)
        save(torch.concat((_x, x_reconstructed), 0).data, f"{sample_path}/2_{batches_done:d}_x_reconstructed.png", n_cols*2, sz=self.config.img_dim)
        save(torch.concat((_x, x_reconstructed_mean), 0).data, f"{sample_path}/3_{batches_done:d}_x_reconstructed_mean.png", n_cols*2, sz=self.config.img_dim)

    # predict funs
    def predict(self, X, *args, score_type="logpx", latent_score_type="normal", probability=False,
            workers=12, batch_size=None, **kwargs):
        """
        model.predict(self, X, *fit_fun_args, 
            score_type="logpx", 
            latent_score_type="normal", 
            probability=False,
            workers=12, 
            batch_size=None, 
            **kwargs):
        """
        # check if we are not requested something which we cannot do
        if probability and self.alpha is None:
            raise ValueError("probability score required, but the alpha coefficients are not fitted, call\n\
                model.fit_alpha(X,y,*args,**kwargs) first")

        # create the dataloader
        loader = self._create_score_loader(X, batch_size=batch_size, workers=workers)

        # get the scores
        basic_score = self._basic_score(loader, score_type, *args, **kwargs)

        # if alpha is not set, do not waste time with computing the latent score
        if not probability:
            return basic_score
            
        # otherwise, compute it and return the sum weighted by alphas
        latent_score = self._latent_score(loader, latent_score_type, *args, **kwargs)
        all_scores = np.concatenate((basic_score.reshape(1,-1), latent_score)).T

        return self.prob_score(all_scores)

    def all_scores(self, X, *args, score_type="logpx", latent_score_type="normal", 
            workers=12, batch_size=None, **kwargs):
        # create the dataloader
        loader = self._create_score_loader(X, batch_size=batch_size, workers=workers)
        
        # get the scores
        basic_score = self._basic_score(loader, score_type, *args, **kwargs)

        # otherwise, compute it and return the sum weighted by alphas
        latent_score = self._latent_score(loader, latent_score_type, *args, **kwargs)

        return np.concatenate((basic_score.reshape(1,-1), latent_score))

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

    def _basic_score(self, loader, score_type, *args, **kwargs):
        if score_type == "logpx":
            return batched_score(self.logpx, loader, self.device, *args, **kwargs)
        else:
            raise ValueError(f'unknown score type {score_type}')

    def _latent_score(self, loader, score_type, *args, **kwargs):
        if score_type == "normal":
            return batched_score(self.normal_latent_score, loader, self.device, *args, **kwargs)
        elif score_type == "kld":
            return batched_score(self.kld_score, loader, self.device, *args, **kwargs)
        elif score_type == "normal_logpx":
            return batched_score(self.normal_logpx_score, loader, self.device, *args, **kwargs)
        else:
            raise ValueError(f'unknown latent score type {latent_score_type}')

    # top-level score
    def logpx(self, x, n=1):
        lpxs = []
        for i in range(n):
            # get the components
            mask, background, foreground = self(x)

            # now get the mean and std
            mu_x = self.compose_image(mask, background, foreground)
            log_var_x = self.log_var_net_x(mu_x)
            std_x = self.std(log_var_x)
            lpx = logpx(x, mu_x, std_x)
            lpxs.append(lpx.data.to('cpu').numpy())

        return -np.mean(lpxs, 0)

    # latent scores
    def normal_latent_score(self, x, n=1):
        scores = []
        for i in range(n):
            zs = self.encoded(x)
            mu = lambda z: torch.zeros(z.shape).to(self.device)
            std = lambda z: torch.ones(z.shape).to(self.device)
            scores.append([logpx(z, mu(z), std(z)).detach().to('cpu').numpy() for z in zs])
            
        return [-np.mean(np.array([s[i] for s in scores]), 0) for i in range(3)]

    def kld_score(self, x, n=1):
        scores = []
        for i in range(n):
            _, kld_s = self._encode(self.vae_shape, x)
            _, kld_b = self._encode(self.vae_background, x)
            _, kld_f = self._encode(self.vae_foreground, x)
            scores.append([kld.detach().to('cpu').numpy() for kld in (kld_s, kld_b, kld_f)])

        return [-np.mean(np.array([s[i] for s in scores]), 0) for i in range(3)]

    def normal_logpx_score(self, x, n=1):
        lpxs = [[],[],[]]
        for i in range(n):
            # get zs and components
            zs = self.encoded(x)
            for i in range(3):
                mzs = [z for z in zs]
                p = torch.distributions.Normal(torch.zeros(zs[i].shape), torch.ones(zs[i].shape))
                mzs[i] = p.rsample().to(self.device)
                mask, background, foreground = self.decode_image_components(mzs)

                # now get the mean and std
                mu_x = self.compose_image(mask, background, foreground)
                log_var_x = self.log_var_net_x(mu_x)
                std_x = self.std(log_var_x)
                lpx = logpx(x, mu_x, std_x)
                lpxs[i].append(lpx.data.to('cpu').numpy())

        return [-np.mean(lpx, 0) for lpx in lpxs]

    # alpha stuff
    def set_alpha(self, alpha):
        if alpha is None:
            self.alpha = None
            return

        if not len(alpha) == 4:
            raise ValueError("given alpha must be a vector of length 4")
        self.alpha = np.array(alpha).reshape(4,)

    def fit_alpha_from_scores(self, scores, y, *args, **kwargs):
        """
        model.fit_alpha(scores, y,
            *args_of_score_funs,
            score_type="logpx",
            latent_score_type="normal",
            workers=12,
            batch_size=None,
            **kwargs_of_score_funs)

        Scores must have shape of (n_samples, 4).
        """
        clf = LogisticRegression(fit_intercept=False).fit(scores, 1-y)
        alpha = clf.coef_[0]
        self.set_alpha(alpha)
        return alpha.reshape(4), clf

    def fit_alpha(self, X, y, *args, **kwargs):
        """
        model.fit_alpha(X, y,
            *args_of_score_funs,
            score_type="logpx",
            latent_score_type="normal",
            workers=12,
            batch_size=None,
            **kwargs_of_score_funs)
        """
        scores = self.all_scores(X, *args, **kwargs).T
        return self.fit_alpha_from_scores(scores, y, *args, **kwargs)

    def save_alpha(self, f):
        np.save(f, self.alpha)

    def load_alpha(self, f):
        alpha = np.load(f)
        self.set_alpha(alpha)

    def prob_score(self, scores):
        ax = np.matmul(scores, self.alpha)
        with warnings.catch_warnings(): # to filter out the ugly exp overflow error
            warnings.simplefilter("ignore")
            return 1/(1+np.exp(ax))

    def cpu_copy(self):
#        device = self.device # save the original device
 #       self.move_to('cpu') # move to cpu
  #      cp = copy.deepcopy(self)
   #     self.move_to(device)
        cp = self
        return cp
        