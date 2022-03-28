from pathlib import Path
import torch
import sgad
import numpy as np
import copy
import time
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN
from tqdm import tqdm
from torch import nn
from torchvision.utils import save_image
import torch.nn.functional as F
import pandas
import os
import time
import random

from sgad.cgn.models.cgn import Reshape, init_net
from sgad.utils import save_cfg, Optimizers, compute_auc, Subset
from sgad.sgvae.utils import Mean, ConvBlock, Encoder, TextureDecoder, ShapeDecoder, rp_trick
from sgad.sgvae.utils import batched_score, logpx

class VAE(nn.Module):
    def __init__(self, 
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
                 log_var_x_estimate = "conv_net",
                 **kwargs
                ):
        """
        VAE constructor
        
        argument values:
        log_var_x_estimate - ["conv_net", "global", "else"] - if else is chosen, supply the 
            model.log_var_net_x with a callable function that returns the log var
        std_approx - ["exp", "softplus"] - how std is computed from log_var
        vae_type - ["shape", "texture"] - the composition of encoder and decoder is a little different
        """
        # supertype init
        super(VAE, self).__init__()
        
        # set seed
        if init_seed is not None:
            torch.random.manual_seed(init_seed)
        
        # params
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.img_dim = img_dim
        self.img_channels = img_channels
        if not std_approx in ["exp", "softplus"]:
            raise ValueError("std_approx can only be `exp` or `softplus`")
        self.std_approx = std_approx # exp or softplus
        init_sz = img_dim // 4
        
        # encoder + decoder
        self.encoder = Encoder(z_dim, img_channels, h_channels, img_dim) 
        if vae_type == "texture":
            self.out_channels = img_channels
            self.decoder = TextureDecoder(z_dim, self.out_channels+1, h_channels, init_sz)
        elif vae_type == "shape":
            self.out_channels = 1
            self.decoder = ShapeDecoder(z_dim, self.out_channels+1, h_channels, init_sz)
        else:
            raise ValueError(f'vae type {vae_type} unknown, try "shape" or "texture"')

        # mu, log_var estimators
        self.mu_net_z = nn.Linear(z_dim*2, z_dim)
        self.log_var_net_z = nn.Linear(z_dim*2, z_dim)
        self.mu_net_x = nn.Conv2d(self.out_channels+1, self.out_channels, 3, stride=1, padding=1, bias=False)
        # either use a convnet or a trainable scalar for log_var_x
        # but also you can support your own function that uses the output of the last layer of the decoder
        if log_var_x_estimate == "conv_net":
            self.log_var_net_x = nn.Sequential(
                nn.Conv2d(self.out_channels+1, 1, 3, stride=1, padding=1, bias=False),
                Mean(1,2,3),
                Reshape(-1,1,1,1)
                )
            self.log_var_x_global = None
        elif log_var_x_estimate == "global":
            self.log_var_x_global = nn.Parameter(torch.Tensor([-1.0]))
            self.log_var_net_x = lambda x: self.log_var_x_global
        else:
            warnings.warn(f"log_var_x_estimate {log_var_x_estimate} not known, you should set .log_var_net_x with a callable function")

        # initialize the net
        init_net(self, init_type=init_type, init_gain=init_gain)
        
        # Optimizers
        self.opts = Optimizers()
        self.opts.set('vae', self, lr=lr, betas=betas)

        # reset seed
        torch.random.seed()

        # choose device automatically
        self.move_to(device)

        # create config
        self.config = CN()
        self.config.z_dim = z_dim
        self.config.h_channels = h_channels
        self.config.img_dim = img_dim
        self.config.img_channels = img_channels
        self.config.batch_size = batch_size
        self.config.init_type = init_type
        self.config.init_gain = init_gain
        self.config.init_seed = init_seed
        self.config.vae_type = vae_type
        self.config.std_approx = std_approx
        self.config.lr = lr
        self.config.betas = betas   
        self.config.log_var_x_estimate = log_var_x_estimate

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
        losses_all = {'iter': [], 'epoch': [], 'kld': [], 'logpx': [], 'elbo': [], 'auc_val': []}

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

        pbar = tqdm(range(n_epochs))
        niter = 0
        start_time = time.time()
        for epoch in pbar:
            for i, data in enumerate(tr_loader):

                # Data to device
                x = data['ims'].to(self.device)

                # encode data, compute kld
                mu_z, log_var_z = self.encode(x)
                std_z = self.std(log_var_z)
                z = rp_trick(mu_z, std_z)
                kld = torch.mean(self.kld(mu_z, log_var_z))
                
                # decode, compute logpx
                mu_x, log_var_x = self.decode(z)
                std_x = self.std(log_var_x)
                lpx = torch.mean(logpx(x, mu_x, std_x))

                # compute elbo
                elbo = torch.mean(kld - lpx)

                # do a step    
                self.opts.zero_grad(['vae'])
                elbo.backward()
                self.opts.step(['vae'], False) # use zero_grad = false here?
                
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

                # output
                if verb:
                    msg = f"[Batch {i}/{len(tr_loader)}]"
                    msg += ''.join(f"[elbo: {get_val(elbo):.3f}]")
                    msg += ''.join(f"[kld: {get_val(kld):.3f}]")
                    msg += ''.join(f"[logpx: {get_val(lpx):.3f}]")
                    msg += ''.join(f"[auc val: {auc_val:.3f}]")
                    pbar.set_description(msg)

                # Saving
                batches_done = epoch * len(tr_loader) + i
                if save_results:
                    if batches_done % save_iter == 0:
                        print(f"Saving samples and weights to {model_path}")
                        self.save_sample_images(x_sample, sample_path, batches_done, n_cols=3)
                        self.save_weights(f"{weights_path}/vae_{batches_done:d}.pth")
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
                    best_model = self.cpu_copy()
                    best_epoch = epoch+1
                    best_auc_val = auc_val
                self.train()

            # exit if running for too long
            if run_time > max_train_time:
                print("Given runtime exceeded, ending training prematurely.")
                break
            
        # finally return self copy if we did not track validation performance 
        if X_val is None:
            best_model = self.cpu_copy()
            best_model.eval()
            best_epoch = n_epochs

        return losses_all, best_model, best_epoch
        
    def std(self, log_var):
        if self.std_approx == "exp":
            return torch.exp(log_var/2)
        else:
            return torch.nn.Softplus(log_var/2) + np.float32(1e-6)
    
    def encode(self, x):
        """Return the mean and log_var vectors of encodings in z space."""
        h = self.encoder(x)
        return self.mu_net_z(h), self.log_var_net_z(h)

    def encoded(self, x):
        """Returns the sampled encoded vectors in z space."""
        mu_z, log_var_z = self.encode(x)
        std_z = self.std(log_var_z)
        return rp_trick(mu_z, std_z)
    
    def decode(self, z):
        """Returns the mean and log_var decodings in x space."""
        h = self.decoder(z)
        return self.mu_net_x(h), self.log_var_net_x(h)
        
    def decoded(self, z):
        """Returns the sampled decodings in x space."""
        mu_x, log_var_x = self.decode(z)
        std_z = self.std(log_var_x)
        return rp_trick(mu_x, std_z)

    def reconstruct(self, x):
        return self.decoded(self.encoded(x))

    def reconstruct_mean(self, x):
        mu_z, log_var_z = self.encode(x)
        z = rp_trick(mu_z, self.std(log_var_z))
        mu_x, log_var_x = self.decode(z)
        return mu_x
    
    def forward(self, x):
        return self.reconstruct(x)
    
    def kld(self, mu, log_var):
        """here input mu_z, log_var_z"""
        return (torch.exp(log_var) + mu.pow(2) - log_var - 1.0).sum(1)/2
                
    def elbo(self, x):
        # first propagate everything
        mu_z, log_var_z = self.encode(x)
        std_z = self.std(log_var_z)
        z = rp_trick(mu_z, std_z)
        
        mu_x, log_var_x = self.decode(z)
        std_x = self.std(log_var_x)
        
        # compute kld
        kl = self.kld(mu_z, log_var_z)
        
        # compute logpx
        lpx = logpx(x, mu_x, std_x)
        
        return kl - lpx

    def generate(self, n):
        p = torch.distributions.Normal(torch.zeros(n, self.z_dim), 1.0)
        z = p.sample().to(self.device)
        mu_x, log_var_x = self.decode(z)
        return rp_trick(mu_x, self.std(log_var_x))

    def generate_mean(self, n):
        p = torch.distributions.Normal(torch.zeros(n, self.z_dim), 1.0)
        z = p.sample().to(self.device)
        return self.decode(z)[0]
        
    def move_to(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        # push to device and train
        self.device = device
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
        x_reconstructed = self.reconstruct(_x).to('cpu')
        x_reconstructed_mean = self.reconstruct_mean(_x).to('cpu')
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

    def logpx(self, x, n=1):
        lpxs = []
        for i in range(n):
            z = self.encoded(x)
            mu_x, log_var_x = self.decode(z)
            std_x = self.std(log_var_x)
            lpx = logpx(x, mu_x, std_x)
            lpxs.append(lpx.data.to('cpu').numpy())

        return -np.mean(lpxs, 0)

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
