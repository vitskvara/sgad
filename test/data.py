import unittest
import sgad
import numpy as np
from torch.utils.data import DataLoader
import os

class TestImports(unittest.TestCase):
    def test_imports(self):
        sgad.cgn
        sgad.utils
        sgad.shared

class TestSplit(unittest.TestCase):
    def test_split_seed(self):
        from sgad.utils import train_val_test_inds

        indices = np.array(range(500))
        tr_inds, val_inds, tst_inds = train_val_test_inds(indices)
        inds1 = train_val_test_inds(indices, seed=3)
        inds2 = train_val_test_inds(indices, seed=3)
        for i in range(3):
            self.assertTrue(np.array_equal(inds1[i], inds2[i]))

    def test_load_cifar10(self):
        # basic cifar
        from sgad.utils import load_cifar10
        cifar10_raw = load_cifar10()
        self.assertTrue(cifar10_raw[0].shape==(60000, 3, 32, 32))
        self.assertTrue(len(cifar10_raw[1])==60000)

        # cifar10 dataloader
        from sgad.utils import CIFAR10
        cifar = CIFAR10()
        bs = 13
        tr_loader, val_loader, tst_loader = sgad.utils.dataloader.split_dataset(cifar, seed=2)
        loader = DataLoader(tr_loader, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        batch = next(iter(loader))
        self.assertTrue(batch['ims'].shape==(bs, 3, 32, 32))
        self.assertTrue(len(batch['labels'])==bs)

    def test_get_dataloaders(self):
        from sgad.utils import get_dataloaders
        dl,_ = get_dataloaders('wildlife_MNIST', 32, 12)
        batch = next(iter(dl))
        self.assertTrue(True)

    def test_split_dataset(self):
        from sgad.utils import SVHN2, split_dataset
        svhn2 = SVHN2()
        tr_set, val_set, tst_set = split_dataset(svhn2, seed=2, target_class=0)

        batch_size = 128
        shuffle = True
        workers = 12
        loader = DataLoader(tr_set, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=workers)

        batch = next(iter(loader))
        self.assertTrue(batch['ims'].shape==(batch_size, 3, 32, 32))
        self.assertTrue(len(batch['labels'])==batch_size)

        sgad.utils.save_resize_img(batch['ims'], "test.png", 8)
        self.assertTrue(os.path.exists("test.png"))

if __name__ == '__main__':
    unittest.main()
