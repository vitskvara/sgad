import torch
from torch import nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from yacs.config import CfgNode as CN
from tqdm import tqdm
import copy
import pandas

import numpy as np
from pathlib import Path
import time
import random
import os, sys

from sgad.sgvae.utils import TextureDecoder, ShapeDecoder, Discriminator, create_score_loader, batched_score
from sgad.utils import save_cfg, Optimizers, Subset, compute_auc

def generator_loss(sg):
    """
    generator_loss(scores_generated)
    
    mean(log(sg)) = E[log(D(G(z)))]
    """
    return - torch.mean(torch.log(sg + 1e-8))

def discriminator_loss(st, sg):
    """
    discriminator_loss(scores_true, scores_generated)
    
    mean(log(st)) + mean(log(1-sg))  = E[log(D(x)) + (log(1-D(G(z))))]
    """
    return - torch.mean(torch.log(st + 1e-8) + torch.log(1 - sg + 1e-8)) / 2.0

def feature_matching_loss(x, xg, discriminator, l):
    """
    Feature-matching loss at the l-th layer of discriminator.
    """
    ht = discriminator[0:l](x)
    hg = discriminator[0:l](xg)
    return nn.MSELoss(reduction='none')(ht, hg).sum((1,2,3))

class GAN(nn.Module):
    def __init__(self, 
                 z_dim=32, 
                 h_channels=32, 
                 img_dim=32, 
                 img_channels=3,
                 n_layers=3,
                 activation="leakyrelu",
                 batch_norm=True,
                 alpha=0.0,
                 fm_depth=7,
                 init_type='orthogonal', 
                 init_gain=0.1, 
                 init_seed=None,
                 batch_size=32, 
                 gan_type="texture", 
                 lr=0.0002,
                 betas=[0.5, 0.999],
                 device=None,
                 **kwargs
                ):
        """
        GAN constructor
        
        alpha - 0.0 - weight of the feature-matching loss
        fm_depth = 7 - after how many layers of the discriminator is the fm loss computed
        gan_type - ["shape", "texture"] - the composition of encoder and decoder is a little different
        """
        # supertype init
        super(GAN, self).__init__()
        
        # set seed
        if init_seed is not None:
            torch.random.manual_seed(init_seed)
        
        # params
        self.z_dim = z_dim
        self.fm_depth = fm_depth
        self.img_dim = img_dim
        self.img_channels = img_channels
        init_sz = img_dim // 4
        self.alpha = alpha
        
        # init generator
        self.out_channels = img_channels
        if gan_type == "texture":
            self.generator = TextureDecoder(z_dim, self.out_channels, h_channels, init_sz, n_layers=n_layers,
                activation=activation, batch_norm=batch_norm)
        if gan_type == "shape":
            self.generator = ShapeDecoder(z_dim, self.out_channels, h_channels, init_sz, n_layers=n_layers,
                activation=activation, batch_norm=batch_norm)
        
        # init discriminator
        self.discriminator = Discriminator(self.out_channels, h_channels, img_dim, n_layers=n_layers,
            activation=activation, batch_norm=batch_norm)
        
        # Optimizers
        self.opts = Optimizers()
        self.opts.set('generator', self.generator, lr=lr, betas=betas)
        self.opts.set('discriminator', self.discriminator, lr=lr, betas=betas)

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
        self.config.activation=activation
        self.config.n_layers = n_layers
        self.config.batch_norm = batch_norm
        self.config.batch_size = batch_size
        self.config.init_type = init_type
        self.config.init_gain = init_gain
        self.config.init_seed = init_seed
        self.config.gan_type = gan_type
        self.config.alpha = alpha
        self.config.fm_depth = fm_depth
        self.config.lr = lr
        self.config.betas = betas   

    def fit(self, X,
            X_val=None, y_val=None,
            n_epochs=50, 
            save_iter=1000, 
            verb=True, 
            save_results=True, 
            save_path=None, 
            save_weights=False,
            workers=12,
            max_train_time=np.inf # in seconds           
           ):
        """Fit the model given X (and possibly y).

        Returns (losses_all, best_model, best_epoch)
        """
        # setup the dataloader
        y = torch.zeros(X.shape[0]).long()
        tr_loader = DataLoader(Subset(torch.tensor(X).float(), y), 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=workers)

        # also check the validation data
        if X_val is not None and y_val is None:
            raise ValueError("X_val given without y_val - please provide it as well.")
        auc_val = best_auc_val = -1.0
        
        # loss values
        losses_all = {'iter': [], 'epoch': [], 'discloss': [], 'genloss': [], 'fmloss': [], 'auc_val': []}

        # setup save paths
        
        if save_path is not None:
            save_results = True
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
        else:
            save_results = False

        pbar = tqdm(range(n_epochs))
        niter = 0
        start_time = time.time()
        for epoch in pbar:
            for i, data in enumerate(tr_loader):
                # true and generated data
                xt = data['ims'].to(self.device)
                xg = self.generate(xt.shape[0])
                
                # optimize the generator
                self.opts.zero_grad(['generator'])
                sg = self.discriminator(xg)
                fml = torch.mean(feature_matching_loss(xt, xg, self.discriminator, self.fm_depth))
                tgl = generator_loss(sg)
                gl = tgl + self.alpha*fml
                gl.backward()
                self.opts.step(['generator'], False)

                # update the discriminator
                self.opts.zero_grad(['discriminator'])
                sgd = self.discriminator(xg.detach())
                st = self.discriminator(xt)
                dl = discriminator_loss(st, sgd)
                dl.backward()
                self.opts.step(['discriminator'], False)
                
                # collect losses
                def get_val(t):
                    return float(t.data.cpu().numpy())
                niter += 1
                losses_all['iter'].append(niter)
                losses_all['epoch'].append(epoch)
                losses_all['discloss'].append(get_val(dl))
                losses_all['genloss'].append(get_val(tgl))
                losses_all['fmloss'].append(get_val(fml))
                losses_all['auc_val'].append(auc_val)

                # output
                if verb:
                    msg = f"[Batch {i}/{len(tr_loader)}]"
                    msg += ''.join(f"[disc: {get_val(dl):.3f}]")
                    msg += ''.join(f"[gen: {get_val(tgl):.3f}]")
                    msg += ''.join(f"[fml: {get_val(fml):.3f}]")
                    msg += ''.join(f"[auc val: {auc_val:.3f}]")
                    pbar.set_description(msg)

                # Saving
                batches_done = epoch * len(tr_loader) + i
                if save_results:
                    if batches_done % save_iter == 0:
                        print(f"Saving samples and weights to {model_path}")
                        self.save_sample_images(x_sample, sample_path, batches_done, n_cols=3)
                        if save_weights:
                            self.save_weights(f"{weights_path}/{batches_done:d}.pth")
                        outdf = pandas.DataFrame.from_dict(losses_all)
                        outdf.to_csv(os.path.join(model_path, "losses.csv"), index=False)

                # exit if running for too long
                run_time = time.time() - start_time
                if run_time > max_train_time:
                    break

            # after every epoch, print the val auc
            if X_val is not None:
                self.eval()
                scores_val = self.predict(X_val, num_workers=workers)
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

    def predict(self, X, workers=12, batch_size=None, **kwargs):
        loader = create_score_loader(X, batch_size if batch_size is not None else self.config.batch_size, 
            workers=workers, shuffle=False)
        return batched_score(lambda x: 1 - self.discriminate(x).detach().to('cpu').numpy(), loader, self.device)

    def generate(self, n):
        p = torch.distributions.Normal(torch.zeros(n, self.z_dim), 1.0)
        z = p.sample().to(self.device)
        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x).reshape(-1)

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

        # push to device
        self.device = device
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        
    def save_sample_images(self, x, sample_path, batches_done, n_cols=3):
        """Saves a grid of generated digits"""
        x_gen = [self.generate(10).to('cpu') for _ in range(n_cols)]
        x_gen = torch.concat(x_gen, 0)
        
        def save(x, path, n_cols, sz=64):
            x = F.interpolate(x, (sz, sz))
            save_image(x.data, path, nrow=n_cols, normalize=True, padding=2)

        Path(sample_path).mkdir(parents=True, exist_ok=True)
        save(x_gen.data, f"{sample_path}/0_{batches_done:d}_x_gen.png", n_cols, sz=self.config.img_dim)
        
    def cpu_copy(self):
        device = self.device
        self.move_to("cpu")
        generator = copy.deepcopy(self.generator)
        discriminator = copy.deepcopy(self.discriminator)
        self.move_to(device)

        cp = GAN(**self.config)
        cp.generator = generator
        cp.discriminator = discriminator
        return cp
