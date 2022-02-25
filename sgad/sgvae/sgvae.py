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

from sgad.cgn.dataloader import Subset
from sgad.utils import Optimizers
from sgad.sgvae import VAE
from sgad.utils import save_cfg, Optimizers, compute_auc, Patch2Image, RandomCrop
from sgad.sgvae.utils import rp_trick, batched_score, logpx
from sgad.shared.losses import BinaryLoss

# Shape-Guided Variational AutoEncoder
class SGVAE(nn.Module):
    """SGVAE(**kwargs)
    
    kwargs = 
        z_dim=32, 
        h_channels=32, 
        img_dim=32, 
        img_channels=3,
        lambda_mask=0.3, 
        weight_mask=100.0,
        init_type='orthogonal', 
        init_gain=0.1, 
        init_seed=None,
        batch_size=1, 
        std_approx="exp",
        lr=0.0002,
        betas=[0.5, 0.999],
        device=None
    """
    def __init__(self, lambda_mask=0.3, weight_mask=100.0, std_approx="exp", device=None, **kwargs):
        # supertype init
        super(SGVAE, self).__init__()
                
        # vaes
        self.vae_shape = VAE(vae_type="shape", std_approx=std_approx, **kwargs)
        self.vae_background = VAE(vae_type="texture", std_approx=std_approx, **kwargs)
        self.vae_foreground = VAE(vae_type="texture", std_approx=std_approx, **kwargs)
        
        # config
        self.config = copy.deepcopy(self.vae_background.config)
        self.device = self.vae_shape.device
        self.config.lambda_mask = lambda_mask
        self.config.weight_mask = weight_mask
        self.z_dim = self.config.z_dim
        self.config.pop('vae_type')

        # binary loss
        self.binary_loss = BinaryLoss(lambda_mask)
        
        # shuffler
        img_dim = self.config.img_dim
        self.shuffler = nn.Sequential(Patch2Image(img_dim, 2), RandomCrop(img_dim))

        # x log var
        self.log_var_x_global = nn.Parameter(torch.Tensor([-1.0])) # nn.Parameter(torch.Tensor([0.0]))
        self.log_var_net_x = lambda x: self.log_var_x_global

        # optimizer
        self.opts = Optimizers()
        self.opts.set('sgvae', self, lr=self.config.lr, betas=self.config.betas)        
        
        # move to device
        self.move_to(device)
        
    def fit(self, X, y=None,
            X_val=None, y_val=None,
            n_epochs=20, 
            save_iter=1000, 
            verb=True, 
            save_results=True, 
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
        
        # loss values
        losses_all = {'iter': [], 'epoch': [], 'kld': [], 'logpx': [], 'elbo': [], 'auc_val': [],
                     'kld_shape': [], 'kld_background': [], 'kld_foreground': [], 'binary': []}

        # setup save paths
        if save_results and save_path == None:
            raise ValueError('If you want to save results, provide the save_path argument.')   
        if save_results:
            model_path = Path(save_path)
            weights_path = model_path / 'weights'
            sample_path = model_path / 'samples'
            weights_path.mkdir(parents=True, exist_ok=True)
            sample_path.mkdir(parents=True, exist_ok=True)

            # dump config
            cfg = copy.deepcopy(self.config)
            cfg.n_epochs = n_epochs
            cfg.save_iter = save_iter
            cfg.save_path = save_path
            cfg.workers = workers
            save_cfg(cfg, model_path / "cfg.yaml")

            # samples for reconstruction
            x_sample = X[random.sample(range(X.shape[0]), 12),:,:,:]

        # tracking
        pbar = tqdm(range(n_epochs))
        niter = 0
        start_time = time.time()
        for epoch in pbar:
            for i, data in enumerate(tr_loader):
                # Data to device
                x = data['ims'].to(self.device)
                
                # now get the klds and zs
                z_s, kld_s = self._encode(self.vae_shape, x)
                z_b, kld_b = self._encode(self.vae_background, x)
                z_f, kld_f = self._encode(self.vae_foreground, x)

                # now get the decoded outputs
                mask = self._decode(self.vae_shape, z_s)
                mask = torch.clamp(mask, 0.0001, 0.9999).repeat(1, self.config.img_channels, 1, 1)
                background = self._decode(self.vae_background, z_b)
                foreground = self._decode(self.vae_foreground, z_f)
                
                # shuffle
                background = self.shuffler(background)
                foreground = self.shuffler(foreground)
                
                # merge them together
                x_hat = mask * foreground + (1 - mask) * background
                mu_x = x_hat
                log_var_x = self.log_var_net_x(x_hat)
                std_x = self.std(log_var_x)
                lpx = torch.mean(logpx(x, mu_x, std_x))
                
                # get elbo
                kld = kld_s + kld_b + kld_f
                elbo = torch.mean(kld - lpx)
                
                # get binary loss
                bin_l = self.binary_loss(mask)
                
                # do a step    
                self.opts.zero_grad(['sgvae'])
                l = elbo + self.config.weight_mask*bin_l
                l.backward()
                self.opts.step(['sgvae'], False) 
                
                # collect losses
                def get_val(t):
                    return float(t.data.cpu().numpy())
                niter += 1
                losses_all['iter'].append(niter)
                losses_all['epoch'].append(epoch)
                losses_all['elbo'].append(get_val(elbo))
                losses_all['kld'].append(get_val(kld))
                losses_all['logpx'].append(get_val(lpx))
                losses_all['auc_val'].append(auc_val)
                losses_all['kld_shape'].append(get_val(kld_s))
                losses_all['kld_background'].append(get_val(kld_b))
                losses_all['kld_foreground'].append(get_val(kld_f))
                losses_all['binary'].append(get_val(bin_l))

                # output
                if verb:
                    msg = f"[Batch {i}/{len(tr_loader)}]"
                    msg += ''.join(f"[elbo: {get_val(elbo):.3f}]")
                    msg += ''.join(f"[kld: {get_val(kld):.3f}]")
                    msg += ''.join(f"[logpx: {get_val(lpx):.3f}]")
                    msg += ''.join(f"[binary: {get_val(bin_l):.3f}]")
                    msg += ''.join(f"[auc val: {auc_val:.3f}]")
                    pbar.set_description(msg)

                # Saving
                batches_done = epoch * len(tr_loader) + i
                if save_results:
                    if batches_done % save_iter == 0:
                        print(f"Saving samples and weights to {model_path}")
                        self.save_sample_images(x_sample, sample_path, batches_done, n_cols=3)
                        self.save_weights(f"{weights_path}/sgvae_{batches_done:d}.pth")
                        outdf = pandas.DataFrame.from_dict(losses_all)
                        outdf.to_csv(os.path.join(model_path, "losses.csv"), index=False)

                # exit if running for too long
                run_time = time.time() - start_time
                if run_time > max_train_time:
                    break

            # after every epoch, print the val auc
            if X_val is not None:
                self.eval()
                scores_val = self.predict(X_val, score_type='logpx', num_workers=workers, n=10)
                auc_val = compute_auc(y_val, scores_val)
                # also copy the params
                if auc_val > best_auc_val:
                    #best_model = self.cpu_copy()
                    best_model = None
                    best_epoch = epoch+1
                    best_auc_val = auc_val
                self.train()

            # exit if running for too long
            if run_time > max_train_time:
                print("Given runtime exceeded, ending training prematurely.")
                break

        best_model = None
        best_epoch = None
        return losses_all, best_model, best_epoch

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
        kld = torch.mean(vae.kld(mu_z, log_var_z))
        return z, kld
    
    def _decode(self, vae, z):
        """Returns mean decoded x for given vae."""
        mu_x, log_var_x = vae.decode(z)
        return mu_x
    
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

    def decode_image_components(self, zs, shuffle=False):
        """Using the means in x space produced by vaes, return the individual image components."""
        (mask, _), (background, _), (foreground, _) = self.decode(zs)
        mask = torch.clamp(mask, 0.0001, 0.9999).repeat(1, self.config.img_channels, 1, 1)
        if shuffle:
            background = self.shuffler(background)
            foreground = self.shuffler(foreground)

        return mask, background, foreground

    def decode_image(self, zs, shuffle=False):
        """For given set of z space encodings, returns a sampled final image."""
        mask, background, foreground = self.decode_image_components(zs, shuffle=shuffle)

        # now get he man and std and sample
        mu_x = mask * foreground + (1 - mask) * background
        log_var_x = self.log_var_net_x(mu_x)
        std_x = self.std(log_var_x)
        return rp_trick(mu_x, std_x)

    def decode_image_mean(self, zs, shuffle=False):
        """For given set of z space encodings, returns a mean final image."""
        mask, background, foreground = self.decode_image_components(zs, shuffle=shuffle)
        return mask * foreground + (1 - mask) * background

    def forward(self, x, shuffle=False):
        """Returns clamped mask and foreground and background."""                
        return self.decode_image_components(self.encoded(x), shuffle=shuffle)

    def reconstruct(self, x, shuffle=False):
        """Returns sampled reconstruction of x."""
        mask, foreground, background = self(x, shuffle=shuffle)        
        mu_x = mask * foreground + (1 - mask) * background
        log_var_x = self.log_var_net_x(mu_x)
        std_x = self.std(log_var_x)
        return rp_trick(mu_x, std_x)

    def reconstruct_mean(self, x, shuffle=False):
        """Returns mean reconstruction of x."""
        mask, foreground, background = self(x, shuffle=shuffle)        
        return mask * foreground + (1 - mask) * background

    def generate(self, n, shuffle=False):
        p = torch.distributions.Normal(torch.zeros(n, self.z_dim), 1.0)
        zs = [p.sample().to(self.device) for _ in range(3)]
        return self.decode_image(zs, shuffle=shuffle)

    def generate_mean(self, n, shuffle=False):
        p = torch.distributions.Normal(torch.zeros(n, self.z_dim), 1.0)
        zs = [p.sample().to(self.device) for _ in range(3)]
        return self.decode_image_mean(zs, shuffle=shuffle)
    
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
        _x = torch.tensor(x).to(self.device)
        mask, foreground, background = self(_x)
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

    def predict(self, X, *args, score_type="logpx", workers=12, **kwargs):
        if not score_type in ["logpx"]:
            raise ValueError("Must be one of ['logpx'].")
        
        # create the loader
        y = torch.zeros(X.shape[0]).long()
        loader = DataLoader(Subset(torch.tensor(X).float(), y), 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=workers)
        
        # get the scores
        if score_type == "logpx":
            return batched_score(self.logpx, loader, self.device, *args, **kwargs)
        else:
            raise ValueError("unknown score type")

    def logpx(self, x, n=1, shuffle=False):
        lpxs = []
        for i in range(n):
            # get zs and components
            zs = self.encoded(x)
            mask, background, foreground = self.decode_image_components(zs, shuffle=shuffle)

            # now get the mean and std
            mu_x = mask * foreground + (1 - mask) * background
            log_var_x = self.log_var_net_x(mu_x)
            std_x = self.std(log_var_x)
            lpx = logpx(x, mu_x, std_x)
            lpxs.append(lpx.data.to('cpu').numpy())

        return np.mean(lpxs, 0)

# TODO
"""
    def cpu_copy(self):
        device = self.device # save the original device
        self.move_to('cpu') # move to cpu
        encoder = copy.deepcopy(self.encoder)
        decoder = copy.deepcopy(self.decoder)
        mu_net_z = copy.deepcopy(self.mu_net_z)
        log_var_net_z = copy.deepcopy(self.log_var_net_z)
        mu_net_x = copy.deepcopy(self.mu_net_x)
        log_var_net_x = copy.deepcopy(self.log_var_net_x)
        log_var_x_global = copy.deepcopy(self.log_var_x_global)
        
        self.move_to(device) # move it back
        cp = VAE( # now create a cpu copy
                z_dim=self.config.z_dim,
                h_channels=self.config.h_channels,
                img_dim=self.config.img_dim,
                img_channels=self.config.img_channels,
                batch_size=self.config.batch_size,
                init_type=self.config.init_type, 
                init_gain=self.config.init_gain,
                init_seed=self.config.init_seed,
                vae_type=self.config.vae_type, 
                std_approx=self.config.std_approx,
                lr=self.config.lr,
                betas=self.config.betas,
                device='cpu',
                log_var_x_estimate=self.config.log_var_x_estimate
                )

        # now replace the parts
        cp.encoder = encoder
        cp.decoder = decoder
        cp.mu_net_z = mu_net_z
        cp.log_var_net_z = log_var_net_z
        cp.mu_net_x = mu_net_x
        cp.log_var_net_x = log_var_net_x
        cp.log_var_x_global = log_var_x_global
        
        return cp
"""