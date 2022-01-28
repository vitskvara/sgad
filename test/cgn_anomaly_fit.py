import unittest
import sgad
import numpy as np
from torch.utils.data import DataLoader
import os
import torch 

from sgad.utils import load_cifar10
from sgad.cgn import Subset
from sgad.cgn.models import CGNAnomaly

X_raw, y_raw = load_cifar10()

class TestFit(unittest.TestCase):
    def test_fit(self):
        X = X_raw[:5000]
        model = CGNAnomaly(batch_size=32)
        losses_all = model.fit(
            X, 
            n_epochs=1, 
            save_iter=100, 
            verb=True, 
            save_results=True, 
            save_path="./_cgn_anomaly_tmp", 
            workers=12
        )

#y_gen = torch.randint(model.config.n_classes, (5,)).long().to(model.device)
#mask, foreground, background = model(y_gen)
#x_gen = mask * foreground + (1 - mask) * background
