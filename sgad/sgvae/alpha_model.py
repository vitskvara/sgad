import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler    

from sgad.utils import compute_auc, Subset

class AlphaClassifier(nn.Module):
    def __init__(self):
        super(AlphaClassifier, self).__init__()
        # Number of input features is 12.
        self.alpha_params = nn.Parameter(torch.Tensor(np.ones(4)))
        self.scaler = StandardScaler()

    def get_alpha(self):
        return F.softmax(self.alpha_params, dim=0)

    def forward(self, x):
        z = torch.Tensor(x)
        z = z * self.get_alpha()
        z = z.sum(1)
        return z
    
    def predict_prob(self, x):
        return torch.sigmoid(self(x))

    def predict(self, x):
        return torch.round(self.predict_prob(x))

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
        
    def fit(self, x, y, tst_x=None, tst_y=None, nepochs = 200, batch_size=256, lr=0.001, negative_p=1,
            workers=1, balanced=True, verb=True, scale=False, early_stopping=False, patience=10):
        if scale:
            self.scaler_fit(x)
            x = self.scaler_transform(x)
            tst_x = tst_x if tst_x is None else self.scaler_transform(tst_x)
            
        # oversample negative data
        X = np.concatenate((x[y == 0].repeat(negative_p, 0), x[y == 0]), 0)
        Y = np.concatenate((y[y == 0].repeat(negative_p, 0), y[y == 0]), 0)
        
        loader = DataLoader(Subset(torch.tensor(X).float(), torch.Tensor(Y)), 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=workers)
        p = (len(y)-sum(y))/sum(y) if balanced else 1.0
        criterion = nn.BCEWithLogitsLoss(torch.Tensor([p]))
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        
        best_auc = self.auc(x, y)
        pat = 0
        for epoch in range(nepochs):
            # output
            if verb:
                crit = criterion(self(torch.Tensor(x)), torch.Tensor(y)).detach().numpy().round(3)
                auc = self.auc(x, y).round(3)
                acc = self.binary_acc(x, y).round(3)
                tst_auc = np.NAN if tst_x is None else self.auc(tst_x, tst_y).round(3)
                print(epoch, ": ", crit, " ", auc, " ", acc, " ", tst_auc, " ", self.get_alpha().detach().numpy())
                
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
                
                opt.zero_grad()
                pred = self(_x)
                loss = criterion(pred, _y)
                loss.backward()
                opt.step()