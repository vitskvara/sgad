import unittest
import sgad
import numpy as np
from torch.utils.data import DataLoader
import os
import torch 
import shutil

from sgad.utils import load_cifar10
from sgad.cgn import Subset
from sgad.cgn.models import CGNAnomaly

X_raw, y_raw = load_cifar10()

class TestFit(unittest.TestCase):
    def test_fit(self):
        X = X_raw[:5000]
        model = CGNAnomaly(batch_size=32)
        _tmp = "./_cgn_anomaly_tmp"
        losses_all = model.fit(
            X, 
            n_epochs=1, 
            save_iter=100, 
            verb=True, 
            save_results=True, 
            save_path=_tmp, 
            workers=12
        )
        self.assertTrue(os.path.isfile(f"{_tmp}/cfg.yaml"))
        self.assertTrue(os.path.isfile(f"{_tmp}/losses.csv"))
        self.assertTrue(os.path.isdir(f"{_tmp}/samples"))
        self.assertTrue(os.path.isdir(f"{_tmp}/weights"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/0_100_x_gen.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/1_100_mask.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/2_100_foreground.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/samples/3_100_background.png"))
        self.assertTrue(os.path.isfile(f"{_tmp}/weights/cgn_100.pth"))
        self.assertTrue(os.path.isfile(f"{_tmp}/weights/discriminator_100.pth"))
        shutil.rmtree(_tmp)

#y_gen = torch.randint(model.config.n_classes, (5,)).long().to(model.device)
#mask, foreground, background = model(y_gen)
#x_gen = mask * foreground + (1 - mask) * background
