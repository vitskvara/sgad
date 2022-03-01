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
import time

from sgad.shared.losses import BinaryLoss, PerceptualLoss
from sgad.utils import save_cfg, Optimizers, compute_auc
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
                img_channels=3,
                disc_model='linear',
                disc_h_dim=32,
                batch_size=32,
                init_type='orthogonal', 
                init_gain=0.1,
                init_seed=None,
                lambda_mask=0.3,
                lambdas_perc=[0.01, 0.05, 0.0, 0.01],
                lr=0.0002,
                betas=[0.5, 0.999],
                device=None
                ):

        super(CGNAnomaly, self).__init__()

        # set seed
        if init_seed is not None:
            torch.random.manual_seed(init_seed)

        # init cgn (generators)
        self.cgn = CGN(
            n_classes=n_classes, 
            latent_sz=z_dim,
            ngf=h_channels, 
            init_type=init_type,
            init_gain=init_gain,
            img_sz=img_dim,
            img_channels=img_channels,
            batch_size=batch_size) # where is the part that accumulates the gradients if we use 1 here?
        
        # init discriminator
        img_shape = [img_channels, img_dim, img_dim]
        Discriminator = DiscLin if disc_model == 'linear' else DiscConv
        self.discriminator = Discriminator(n_classes=n_classes, ndf=disc_h_dim, img_shape=img_shape)

        # Loss functions
        self.adv_loss = torch.nn.MSELoss() # ?
        self.binary_loss = BinaryLoss(lambda_mask) # this enforces the binary mask
        self.perc_loss = PerceptualLoss(style_wgts=lambdas_perc) # similarity of pictures 

        # Optimizers
        self.opts = Optimizers()
        self.opts.set('generator', self.cgn, lr=lr, betas=betas)
        self.opts.set('discriminator', self.discriminator, lr=lr, betas=betas)

        # reset seed
        torch.random.seed()
        
        # choose device automatically
        self.move_to(device)

        # create config
        self.config = CN()
        self.config.z_dim = z_dim
        self.config.h_channels = h_channels
        self.config.n_classes = n_classes
        self.config.img_dim = img_dim
        self.config.img_channels = img_channels
        self.config.disc_model = disc_model
        self.config.disc_h_dim = disc_h_dim
        self.config.batch_size = batch_size
        self.config.init_type = init_type
        self.config.init_gain = init_gain
        self.config.init_seed = init_seed
        self.config.lambda_mask = lambda_mask
        self.config.lambdas_perc = lambdas_perc
        self.config.lr = lr
        self.config.betas = betas
                
    def get_inp(self, ys):
        return self.cgn.get_inp(ys)

    def forward(self, ys=None, counterfactual=False):
        return self.cgn.forward(ys, counterfactual)

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
        losses_all = {'iter': [], 'epoch': [], 'g_adv': [], 'g_binary': [], 'g_perc': [], 
            'd_real': [], 'd_fake': [], 'auc_val': []}

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
        start_time = time.time()
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
                    return float(t.data.cpu().numpy())
                niter += 1
                losses_all['iter'].append(niter)
                losses_all['epoch'].append(epoch)
                losses_all['g_adv'].append(get_val(losses_g['adv']))
                losses_all['g_binary'].append(get_val(losses_g['binary'].data))
                losses_all['g_perc'].append(get_val(losses_g['perc'].data))
                losses_all['d_real'].append(get_val(losses_d['real'].data))
                losses_all['d_fake'].append(get_val(losses_d['fake'].data))
                losses_all['auc_val'].append(auc_val)

                # output
                if verb:
                    msg = f"[Batch {i}/{len(tr_loader)}]"
                    msg += ''.join([f"[{k}: {v:.3f}]" for k, v in losses_d.items()])
                    msg += ''.join([f"[{k}: {v:.3f}]" for k, v in losses_g.items()])
                    msg += ''.join(f"[auc val: {auc_val:.3f}]")
                    pbar.set_description(msg)

                # Saving
                batches_done = epoch * len(tr_loader) + i
                if save_results:
                    if batches_done % save_iter == 0:
                        print(f"Saving samples and weights to {model_path}")
                        self.save_sample_images(sample_path, batches_done, n_rows=3)
                        self.save_weights(
                            f"{weights_path}/cgn_{batches_done:d}.pth",
                            f"{weights_path}/discriminator_{batches_done:d}.pth")
                        outdf = pandas.DataFrame.from_dict(losses_all)
                        outdf.to_csv(os.path.join(model_path, "losses.csv"), index=False)

                # exit if running for too long
                run_time = time.time() - start_time
                if run_time > max_train_time:
                    break

            # after every epoch, print the val auc
            if X_val is not None:
                self.eval()
                scores_val = self.predict(X_val)
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
        save(x_gen.data, f"{sample_path}/0_{batches_done:d}_x_gen.png", n_rows, sz=self.config.img_dim)
        save(mask.data, f"{sample_path}/1_{batches_done:d}_mask.png", n_rows, sz=self.config.img_dim)
        save(foreground.data, f"{sample_path}/2_{batches_done:d}_foreground.png", n_rows, sz=self.config.img_dim)
        save(background.data, f"{sample_path}/3_{batches_done:d}_background.png", n_rows, sz=self.config.img_dim)

    def predict(self, X, score_type="discriminator", workers=12):
        if not score_type in ["discriminator", "perceptual"]:
            raise ValueError("Must be one of ['discriminator', 'perceptual'].")
        
        # create the loader
        y = torch.zeros(X.shape[0]).long()
        loader = DataLoader(Subset(torch.tensor(X).float(), y), 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=workers)
        
        # get the scores
        if score_type == "discriminator":
            return self.disc_score_batched(loader)
        else:
            return self.perc_score_batched(loader)

    def disc_score(self, X):
        y = torch.zeros(X.size(0),).int().to(self.device)
        return 1 - self.discriminator(X, y).data.to('cpu').numpy()

    def disc_score_batched(self, loader):
        scores = []
        labels = []
        for batch in loader:
            x = batch['ims'].to(self.device)
            score = self.disc_score(x)
            scores.append(score)

        return np.concatenate(scores)

    def perc_score(self, X):
        x_gen = self.generate_random(X.size(0))
        x_gen = x_gen.detach().to(self.device)
        return np.array([self.perc_loss(x_gen[i].reshape(1,*x_gen[i].size()), X[i].reshape(1,*X[i].size())).data.to('cpu').numpy() for i in range(X.size(0))])

    def perc_score_batched(self, loader):
        scores = []
        for batch in loader:
            x = batch['ims'].to(self.device)
            score = self.perc_score(x)
            scores.append(score)

        return np.concatenate(scores)

    def num_params(self):
        s = 0
        for p in self.parameters():
            s += np.array(p.data.to('cpu')).size
        return s

    def save_weights(self, cgn_file, disc_file):
        torch.save(self.cgn.state_dict(), cgn_file)
        torch.save(self.discriminator.state_dict(), disc_file)

    def move_to(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        # push to device and train
        self.device = device
        self.cgn = self.cgn.to(device)
        self.discriminator = self.discriminator.to(device)
        self.adv_loss = self.adv_loss.to(device)
        self.binary_loss = self.binary_loss.to(device)
        self.perc_loss = self.perc_loss.to(device)

    def cpu_copy(self):
        device = self.device # save the original device
        self.move_to('cpu') # move to cpu
        cgn = copy.deepcopy(self.cgn)
        disc = copy.deepcopy(self.discriminator)
        self.move_to(device) # move it back
        cp = CGNAnomaly( # now create a cpu copy
                z_dim=self.config.z_dim,
                h_channels=self.config.h_channels,
                n_classes=self.config.n_classes,
                img_dim=self.config.img_dim,
                img_channels=self.config.img_channels,
                disc_model=self.config.disc_model,
                disc_h_dim=self.config.disc_h_dim,
                batch_size=self.config.batch_size,
                init_type=self.config.init_type, 
                init_gain=self.config.init_gain,
                init_seed=self.config.init_seed,
                lambda_mask=self.config.lambda_mask,
                lambdas_perc=self.config.lambdas_perc,
                lr=self.config.lr,
                betas=self.config.betas,
                device='cpu')
        cp.cgn = cgn
        cp.discriminator = disc
        return cp
