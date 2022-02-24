import torch
import numpy as np
from torch import nn
import copy
from tqdm import tqdm
from pathlib import Path
import time
import random
from torch.utils.data import DataLoader

from sgad.cgn.dataloader import Subset
from sgad.utils import Optimizers
from .vae import VAE
from sgad.utils import save_cfg, Optimizers, compute_auc, Patch2Image, RandomCrop
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
        self.opts.set('cvae', self, lr=self.config.lr, betas=self.config.betas)        
        
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
            x_sample = X[random.sample(range(X.shape[0]), 30),:,:,:]

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
                logpx = torch.mean(self.logpx(x, mu_x, log_var_x))
                
                # get elbo
                kld = kld_s + kld_b + kld_f
                elbo = torch.mean(kld - logpx)
                
                # get binary loss
                bin_l = self.binary_loss(mask)
                
                # do a step    
                self.opts.zero_grad(['cvae'])
                l = elbo + self.config.weight_mask*bin_l
                l.backward()
                self.opts.step(['cvae'], False) 
                
                # collect losses
                def get_val(t):
                    return float(t.data.cpu().numpy())
                niter += 1
                losses_all['iter'].append(niter)
                losses_all['epoch'].append(epoch)
                losses_all['elbo'].append(get_val(elbo))
                losses_all['kld'].append(get_val(kld))
                losses_all['logpx'].append(get_val(logpx))
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
                    msg += ''.join(f"[logpx: {get_val(logpx):.3f}]")
                    msg += ''.join(f"[binary: {get_val(bin_l):.3f}]")
                    msg += ''.join(f"[auc val: {auc_val:.3f}]")
                    pbar.set_description(msg)
                    
        best_model = None
        best_epoch = None
        return losses_all, best_model, best_epoch

    def std(self, log_var):
        if self.config.std_approx == "exp":
            return torch.exp(log_var/2)
        else:
            return torch.nn.Softplus(log_var/2) + np.float32(1e-6)
   
    def logpx(self, x, mu, log_var):
        """here input mu_x, log_var_x"""
        p = torch.distributions.Normal(mu, self.std(log_var))
        return p.log_prob(x).sum((1,2,3))
    
    def _encode(self, vae, x):
        mu_z, log_var_z = vae.encode(x)
        std_z = vae.std(log_var_z)
        z = vae.rp_trick(mu_z, std_z)
        kld = torch.mean(vae.kld(mu_z, log_var_z))
        return z, kld
    
    def _decode(self, vae, z):
        mu_x, log_var_x = vae.decode(z)
        return mu_x
    
    def encode(self, x):
        encodings_s = self.vae_shape.encode(x)
        encodings_b = self.vae_background.encode(x)
        encodings_f = self.vae_foreground.encode(x)
        return encodings_s, encodings_b, encodings_f
    
    def encoded(self, x):
        encodings_s = self.vae_shape.encoded(x)
        encodings_b = self.vae_background.encoded(x)
        encodings_f = self.vae_foreground.encoded(x)
        return encodings_s, encodings_b, encodings_f

    def decode(self, zs):
        decodings_s = self.vae_shape.decode(zs[0])
        decodings_b = self.vae_background.decode(zs[1])
        decodings_f = self.vae_foreground.decode(zs[2])
        return decodings_s, decodings_b, decodings_f
    
    def decoded(self, zs):
        decodings_s = self.vae_shape.decoded(zs[0])
        decodings_b = self.vae_background.decoded(zs[1])
        decodings_f = self.vae_foreground.decoded(zs[2])
        return decodings_s, decodings_b, decodings_f
    
    def forward(self, x):
        zs = self.encoded(x)
        (mask, _), (background, _), (foreground, _) = self.decode(zs)
        mask = torch.clamp(mask, 0.0001, 0.9999).repeat(1, self.config.img_channels, 1, 1)
        background = self.shuffler(background)
        foreground = self.shuffler(foreground)
                
        return mask, foreground, background
    
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
