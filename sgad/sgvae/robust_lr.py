import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler    

from sgad.utils import compute_auc, Subset

class RobustLogisticRegression(nn.Module):
    def __init__(self, input_dim, alpha=None, beta=1.0, alpha0=None):
        super(RobustLogisticRegression, self).__init__()
        # set the initial alpha value
        try:
            if alpha == None:
                alpha = np.ones(input_dim)
        except ValueError: # do nothing
            None
        except:
            raise
        self.alpha = nn.Parameter(torch.Tensor(alpha))
        # set alpha prior
        try:
            if alpha0 == None:
                alpha0 = np.concatenate(([1], np.zeros(input_dim-1)))
        except ValueError:
            None
        except:
            raise
        self.alpha0 = torch.Tensor(alpha0)
        self.beta = beta
        self.scaler = StandardScaler()
        self.input_dim = input_dim

    def get_alpha(self):
        return self.alpha

    def forward(self, x):
        t = torch.Tensor(x)
        t = t * self.get_alpha()
        t = t.sum(1)
        return t
    
    def predict_prob(self, x):
        return 1/(1+torch.exp(-self(x)))

    def predict(self, x):
        return torch.round(self.predict_prob(x))
    
    def prior_loss(self):
        return self.beta*torch.sqrt(torch.square(self.alpha0 - self.alpha).sum())

    def binary_acc(self, x, y):
        y_true = np.array(y)
        y_pred = self.predict(x).detach().numpy()
        nhits = (y_pred == y_true).sum()
        return nhits/y_true.shape[0]
    
    def auc(self, x, y):
        scores = self.predict_prob(x)
        return compute_auc(y, scores.detach().numpy())
    
    def scaler_fit(self, x):
        self.scaler.fit(x)
        
    def scaler_transform(self, x):
        return self.scaler.transform(x)
    
    def save_weights(self, f):
        np.save(f, self.alpha_params.detach().numpy())
                
    def fit(self, x, y, tst_x=None, tst_y=None, nepochs = 2, batch_size=256, lr=0.01, opt_type="lbfgs",
            workers=1, balanced=True, verb=True, scale=False, early_stopping=False, patience=10):
        if scale:
            self.scaler_fit(x)
            x = self.scaler_transform(x)
            tst_x = tst_x if tst_x is None else self.scaler_transform(tst_x)
            
        # oversample the data so that both classes have the same number of samples
        if balanced:
            n1 = int(sum(y))
            n0 = len(y) - n1
            if n0 < n1:
                p = int(np.floor(n1/n0))
                X = np.concatenate((x[y == 0].repeat(p, 0), x[y == 1]), 0)
                Y = np.concatenate((y[y == 0].repeat(p, 0), y[y == 1]), 0)
            else:
                p = int(np.floor(n0/n1))
                X = np.concatenate((x[y == 1].repeat(p, 0), x[y == 0]), 0)
                Y = np.concatenate((y[y == 1].repeat(p, 0), y[y == 0]), 0)
        else:
            X = x
            Y = y
        
        # setup loader, optimiser, loss
        loader = DataLoader(Subset(torch.tensor(X).float(), torch.Tensor(Y)), 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=workers)
        criterion = nn.BCEWithLogitsLoss()
        if opt_type == "lbfgs":
            opt = torch.optim.LBFGS(self.parameters(), lr=lr)
        elif opt_type == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            # override the default
            if nepochs == 2:
                nepochs = 200
        else:
            raise ValueError('opt_type must be lbfgs or adam')
        
        # setup stopping criterion
        best_auc = self.auc(x, y)
        pat = 0
        
        # print first line of output
        if verb:
            print("epoch :  tloss   ploss   auc   acc   tst_auc   alpha")
        for epoch in range(nepochs):
            # output
            if verb:
                crit = criterion(self(torch.Tensor(x)), torch.Tensor(y)).detach().numpy().round(3)
                prloss = self.prior_loss().detach().numpy().round(3)
                auc = self.auc(x, y).round(3)
                acc = self.binary_acc(x, y).round(3)
                tst_auc = np.NAN if tst_x is None else self.auc(tst_x, tst_y).round(3)
                _alpha = self.get_alpha().detach().numpy()
                print(epoch, "    : ", crit, " ", prloss, " ", auc, " ", acc, " ", tst_auc, " ", _alpha)
                
            # early stopping
            if early_stopping:
                cur_auc = self.auc(x, y)
                if cur_auc < best_auc:
                    if pat < patience:
                        pat += 1
                    else:
                        if verb:
                            print(f"stopping early after {epoch} epochs")
                        return
                else:
                    best_auc = cur_auc
                    pat = 0
                    
            # training
            for batch in loader:
                _x, _y = batch['ims'], batch['labels']

                def loss_closure():
                    opt.zero_grad()
                    pred = self(_x)
                    loss = criterion(pred, _y) + self.prior_loss()
                    loss.backward()
                    return loss

                if opt_type == "lbfgs":
                    opt.step(loss_closure)
                elif opt_type == "adam":
                    loss_closure()
                    opt.step()
                else:
                    raise ValueError('opt_type must be lbfgs or adam')

        # check for bad values
        if (abs(self.alpha.detach().numpy()) > 1e6).any():
            raise RuntimeError('convergence failed')

