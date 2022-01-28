import torch
import numpy as np
from torch import nn
from pathlib import Path
import copy
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
import pandas
import os

from sgad.shared.losses import BinaryLoss, PerceptualLoss
from sgad.utils import save_cfg, Optimizers
from .cgn import CGN
from .discriminator import DiscLin, DiscConv
from sgad.cgn.dataloader import Subset

class CGNAnomaly(nn.Module):

    def __init__(
                self, 
                z_dim=32,
                h_channels=32,
                n_classes=1,
                img_dim=32,
                disc_model='linear',
                disc_h_dim=32,
                batch_size=1,  # does this need to be here?
                init_type='orthogonal', 
                init_gain=0.1,
                lambda_mask=0.3,
                lambdas_perc=[0.01, 0.05, 0.0, 0.01],
                lr=0.0002,
                betas=[0.5, 0.999],
                ):

        super(CGNAnomaly, self).__init__()

        # init cgn (generators)
        self.cgn = CGN(
            n_classes=n_classes, 
            latent_sz=z_dim,
            ngf=h_channels, 
            init_type=init_type,
            init_gain=init_gain,
            img_sz=img_dim,
            batch_size=batch_size) # where is the part that accumulates the gradients if we use 1 here?
        
        # init discriminator
        Discriminator = DiscLin if disc_model == 'linear' else DiscConv
        self.discriminator = Discriminator(n_classes=n_classes, ndf=disc_h_dim)

        # Loss functions
        self.adv_loss = torch.nn.MSELoss() # ?
        self.binary_loss = BinaryLoss(lambda_mask) # this enforces the binary mask
        self.perc_loss = PerceptualLoss(style_wgts=lambdas_perc) # similarity of pictures 

        # Optimizers
        self.opts = Optimizers()
        self.opts.set('generator', self.cgn, lr=lr, betas=betas)
        self.opts.set('discriminator', self.discriminator, lr=lr, betas=betas)

        # push to device and train
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.cgn = self.cgn.to(device)
        self.discriminator = self.discriminator.to(device)
        self.adv_loss = self.adv_loss.to(device)
        self.binary_loss = self.binary_loss.to(device)
        self.perc_loss = self.perc_loss.to(device)

        # create config
        self.config = CN()
        self.config.z_dim = z_dim
        self.config.h_channels = h_channels
        self.config.n_classes = n_classes
        self.config.img_dim = img_dim
        self.config.disc_model = disc_model
        self.config.disc_h_dim = disc_h_dim
        self.config.batch_size = batch_size
        self.config.init_type = init_type
        self.config.init_gain = init_gain
        self.config.lambda_mask = lambda_mask
        self.config.lambdas_perc = lambdas_perc
        self.config.lr = lr
        self.config.betas = betas
                
    def get_inp(self, ys):
        return self.cgn.get_imp(ys)

    def forward(self, ys=None, counterfactual=False):
        return self.cgn.forward(ys, counterfactual)

    def fit(self, X, y=None,
            n_epochs=20, 
            save_iter=1000, 
            verb=True, 
            save_results=True, 
            save_path=None, 
            workers=12
        ):
        """Fit the model given X (and possibly y).

        If y is supported, the classes should be labeled by integers like in range(n_classes).
        """
        # setup the dataloader
        y = torch.zeros(X.shape[0]).long() if y is None else y
        tr_loader = DataLoader(Subset(torch.tensor(X).float(), y), 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=workers)
        
        # loss values
        losses_all = {'iter': [], 'epoch': [], 'g_adv': [], 'g_binary': [], 'g_perc': [], 
            'd_real': [], 'd_fake': []}

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

        pbar = tqdm(range(n_epochs))
        niter = 0
        for epoch in pbar:
            for i, data in enumerate(tr_loader):

                # Data and adversarial ground truths to device
                x_gt = data['ims'].to(self.device)
                y_gt = data['labels'].to(self.device)
                valid = torch.ones(len(y_gt),).to(self.device)
                fake = torch.zeros(len(y_gt),).to(self.device)

                #
                #  Train Generator
                #
                self.opts.zero_grad(['generator'])

                # Generate some samples
                y_gen = torch.randint(self.config.n_classes, (len(y_gt),)).long().to(self.device)
                mask, foreground, background = self.cgn(y_gen)
                x_gen = mask * foreground + (1 - mask) * background

                # Calc Losses
                validity = self.discriminator(x_gen, y_gen)

                losses_g = {}
                losses_g['adv'] = self.adv_loss(validity, valid)
                losses_g['binary'] = self.binary_loss(mask)
                losses_g['perc'] = self.perc_loss(x_gen, x_gt)

                # Backprop and step
                loss_g = sum(losses_g.values())
                loss_g.backward()
                self.opts.step(['generator'], False)

                #
                # Train Discriminator
                #
                self.opts.zero_grad(['discriminator'])

                # Discriminate real and fake
                validity_real = self.discriminator(x_gt, y_gt)
                validity_fake = self.discriminator(x_gen.detach(), y_gen)

                # Losses
                losses_d = {}
                losses_d['real'] = self.adv_loss(validity_real, valid)
                losses_d['fake'] = self.adv_loss(validity_fake, fake)
                loss_d = sum(losses_d.values()) / 2

                # Backprop and step
                loss_d.backward()
                self.opts.step(['discriminator'], False)

                # collect losses
                def get_val(t):
                    return t.data.cpu().numpy()
                niter += 1
                losses_all['iter'].append(niter)
                losses_all['epoch'].append(epoch)
                losses_all['g_adv'].append(get_val(losses_g['adv']))
                losses_all['g_binary'].append(get_val(losses_g['binary'].data))
                losses_all['g_perc'].append(get_val(losses_g['perc'].data))
                losses_all['d_real'].append(get_val(losses_d['real'].data))
                losses_all['d_fake'].append(get_val(losses_d['fake'].data))

                # output
                if verb:
                    msg = f"[Batch {i}/{len(tr_loader)}]"
                    msg += ''.join([f"[{k}: {v:.3f}]" for k, v in losses_d.items()])
                    msg += ''.join([f"[{k}: {v:.3f}]" for k, v in losses_g.items()])
                    pbar.set_description(msg)

                # Saving
                batches_done = epoch * len(tr_loader) + i
                if save_results:
                    if batches_done % save_iter == 0:
                        print(f"Saving samples and weights to {model_path}")
                        self.save_sample_images(sample_path, batches_done, n_rows=3)
                        torch.save(self.cgn.state_dict(), f"{weights_path}/cgn_{batches_done:d}.pth")
                        torch.save(self.discriminator.state_dict(), f"{weights_path}/discriminator_{batches_done:d}.pth")
                        outdf = pandas.DataFrame.from_dict(losses_all)
                        outdf.to_csv(os.path.join(model_path, "losses.csv"), index=False)
                
        return losses_all

    def generate(self, y):
        mask, foreground, background = self.cgn(y)
        x_gen = mask * foreground + (1 - mask) * background
        return x_gen    

    def generate_random(self, n):
        y = torch.randint(self.config.n_classes, (n,)).long().to(self.device)
        return self.generate(y)

    def save_sample_images(self, sample_path, batches_done, n_rows=3):
        """Saves a grid of generated digits"""
        y_gen = np.arange(self.config.n_classes).repeat(n_rows)
        y_gen = torch.LongTensor(y_gen).to(self.device)
        mask, foreground, background = self(y_gen)
        x_gen = mask * foreground + (1 - mask) * background

        def save(x, path, n_rows, sz=64):
            x = F.interpolate(x, (sz, sz))
            save_image(x.data, path, nrow=n_rows, normalize=True, padding=2)

        Path(sample_path).mkdir(parents=True, exist_ok=True)
        save(x_gen.data, f"{sample_path}/0_{batches_done:d}_x_gen.png", n_rows)
        save(mask.data, f"{sample_path}/1_{batches_done:d}_mask.png", n_rows)
        save(foreground.data, f"{sample_path}/2_{batches_done:d}_foreground.png", n_rows)
        save(background.data, f"{sample_path}/3_{batches_done:d}_background.png", n_rows)

    def predict(self):
        return None
